#!/usr/bin/env python3
"""Probabilistic hourly forecast for a ticker with Kronos-small.

KronosPredictor averages its samples internally when sample_count > 1, which
collapses the distribution into a single path. To get an uncertainty band we
instead run the predictor N separate times with sample_count=1 and take
per-step quantiles over the independent paths. The share of runs that end
above the last close gives the directional call (bullish / bearish).

Usage:
    python forecast.py BTC-USD                   # candles from yfinance
    python forecast.py AAPL --pred-len 14 --runs 50
    python forecast.py XAUUSD --mt5              # candles from a running MetaTrader 5 terminal (Windows)
    python forecast.py XAUUSD --csv xauusd_h1.csv  # MT5 "Export Bars" file or any OHLC CSV

Kronos is imported from ./Kronos (a clone of github.com/shiyu-coder/Kronos);
set KRONOS_DIR to use a clone somewhere else.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from matplotlib.ticker import FuncFormatter

KRONOS_DIR = Path(os.environ.get("KRONOS_DIR", Path(__file__).resolve().parent / "Kronos"))
if not (KRONOS_DIR / "model" / "kronos.py").exists():
    sys.exit(f"Kronos not found at {KRONOS_DIR}; clone https://github.com/shiyu-coder/Kronos there or set KRONOS_DIR")
sys.path.insert(0, str(KRONOS_DIR))

from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MODEL_ID = "NeoQuasar/Kronos-small"

# A run "votes" bullish when its final close is above the last observed close.
# At least this share of runs must agree before we call a direction.
DIRECTION_THRESHOLD = 0.6

# Plot palette: forecast in the primary series blue, history as recessive neutral ink.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
HISTORY = "#8a8984"
FORECAST = "#2a78d6"
BULLISH = "#006300"
BEARISH = "#d03b3b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ticker", nargs="?", help="yfinance ticker, or MT5 symbol with --mt5 (default: BTC-USD)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--mt5", action="store_true",
                     help="read H1 bars from a running MetaTrader 5 terminal (Windows; pip install MetaTrader5)")
    src.add_argument("--csv", type=Path, help="read candles from a CSV (MT5 'Export Bars' format or OHLC columns)")
    p.add_argument("--lookback", type=int, default=400, help="hourly candles fed to the model (default: 400)")
    p.add_argument("--pred-len", type=int, default=24, help="hours to forecast (default: 24)")
    p.add_argument("--runs", type=int, default=20, help="independent predictor runs (default: 20)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature T (default: 1.0)")
    p.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling top_p (default: 0.9)")
    p.add_argument("--seed", type=int, default=0, help="run i is seeded with seed+i (default: 0)")
    p.add_argument("--plot-history", type=int, default=120, help="history candles shown in the plot (default: 120)")
    p.add_argument("--out-dir", type=Path, default=Path("output"), help="where the PNG and CSV go (default: output/)")
    args = p.parse_args()
    if args.ticker is None:
        args.ticker = args.csv.stem if args.csv else "BTC-USD"
    return args


def finish_candles(df: pd.DataFrame, name: str, lookback: int) -> pd.DataFrame:
    """Normalise to float OHLCV columns on a sorted, naive DatetimeIndex and keep the last `lookback` rows."""
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna().astype("float64")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if len(df) < lookback:
        sys.exit(f"only {len(df)} hourly candles available for {name!r}, need {lookback}")
    return df.tail(lookback)


def fetch_yfinance(ticker: str, lookback: int) -> pd.DataFrame:
    """Last `lookback` completed hourly candles from yfinance, in exchange-local time."""
    df = yf.Ticker(ticker).history(period="1y", interval="1h")
    if df.empty:
        sys.exit(f"yfinance returned no hourly data for {ticker!r} (check the symbol on finance.yahoo.com; "
                 f"gold futures are GC=F, spot XAUUSD needs --mt5 or --csv)")

    # Drop the still-forming candle so the model only sees closed bars.
    if df.index[-1] + pd.Timedelta(hours=1) > pd.Timestamp.now(tz=df.index.tz):
        df = df.iloc[:-1]
    return finish_candles(df, ticker, lookback)


def fetch_mt5(symbol: str, lookback: int, timeframe: str = "H1") -> pd.DataFrame:
    """Last `lookback` completed bars (MT5 timeframe name, e.g. H1, M15, D1) from a running
    MetaTrader 5 terminal, in broker server time.

    Uses the account the terminal is logged into, or MT5_LOGIN / MT5_PASSWORD / MT5_SERVER if set.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        sys.exit("--mt5 needs the MetaTrader5 package, which only runs on Windows: pip install MetaTrader5")

    login = {}
    if os.environ.get("MT5_LOGIN"):
        login = {"login": int(os.environ["MT5_LOGIN"]), "password": os.environ.get("MT5_PASSWORD", ""),
                 "server": os.environ.get("MT5_SERVER", "")}
    if not mt5.initialize(**login):
        sys.exit(f"could not connect to the MetaTrader 5 terminal: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol, True):
            sys.exit(f"symbol {symbol!r} not found in MT5 (check the exact name in Market Watch, e.g. XAUUSD or GOLD)")
        # Position 0 is the still-forming bar; start at 1 so only closed candles are used.
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), 1, lookback)
        if rates is None or len(rates) == 0:
            sys.exit(f"MT5 returned no {timeframe} bars for {symbol!r}: {mt5.last_error()}")
    finally:
        mt5.shutdown()

    df = pd.DataFrame(rates)
    df.index = pd.to_datetime(df["time"], unit="s")
    # Spot FX and metals have no exchange volume; tick volume is the usual stand-in.
    df["volume"] = df["real_volume"] if df["real_volume"].any() else df["tick_volume"]
    return finish_candles(df, f"{symbol} {timeframe}", lookback)


def load_csv(path: Path, lookback: int) -> pd.DataFrame:
    """Candles from a CSV: MT5 'Export Bars' files (<DATE> <TIME> <OPEN> ...) or open/high/low/close columns."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]

    if {"date", "time"} <= set(df.columns):
        stamps = pd.to_datetime(df["date"].astype(str).str.replace(".", "-", regex=False) + " " + df["time"].astype(str))
    else:
        col = next((c for c in ("time", "datetime", "timestamp", "timestamps", "date") if c in df.columns), None)
        if col is None:
            sys.exit(f"{path}: no time column found (expected MT5 <DATE>/<TIME> or a time/datetime column)")
        stamps = pd.to_datetime(df[col], unit="s") if pd.api.types.is_numeric_dtype(df[col]) else pd.to_datetime(df[col])
    df.index = pd.DatetimeIndex(stamps)

    if "volume" not in df.columns and "vol" in df.columns:
        df["volume"] = df["vol"]
    tick_col = next((c for c in ("tickvol", "tick_volume") if c in df.columns), None)
    if tick_col and ("volume" not in df.columns or not df["volume"].any()):
        df["volume"] = df[tick_col]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return finish_candles(df, str(path), lookback)


def future_timestamps(index: pd.DatetimeIndex, n: int, step: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.Series:
    """Next `n` bar times, `step` apart.

    When the history covers at least a week, only (weekday, hour, minute) slots
    seen in it are used: every hour for 24/7 markets like crypto, session hours
    on trading days for equities, FX and metals. Shorter histories (e.g. 400
    one-minute bars) can't show the weekly pattern, so they just step forward.
    """
    slots = None
    if index[-1] - index[0] >= pd.Timedelta(days=7):
        slots = set(zip(index.dayofweek, index.hour, index.minute))
    out, t = [], index[-1]
    while len(out) < n:
        t += step
        if slots is None or (t.dayofweek, t.hour, t.minute) in slots:
            out.append(t)
    return pd.Series(out)


def sample_paths(predictor: KronosPredictor, candles: pd.DataFrame, y_timestamp: pd.Series,
                 args: argparse.Namespace, verbose: bool = True) -> np.ndarray:
    """Run the predictor `args.runs` times (sample_count=1 each); returns close paths, shape (runs, pred_len)."""
    x_timestamp = pd.Series(candles.index)
    paths = []
    for i in range(args.runs):
        torch.manual_seed(args.seed + i)
        start = time.perf_counter()
        pred = predictor.predict(
            df=candles,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=args.pred_len,
            T=args.temperature,
            top_p=args.top_p,
            sample_count=1,
            verbose=False,
        )
        paths.append(pred["close"].to_numpy())
        if verbose:
            print(f"  run {i + 1:>2}/{args.runs}: close in {args.pred_len}h = {paths[-1][-1]:,.2f} "
                  f"({time.perf_counter() - start:.1f}s)")
    return np.vstack(paths)


def direction(paths: np.ndarray, last_close: float) -> tuple[str, int]:
    """Directional call from the share of runs whose final close is above `last_close`."""
    up = int((paths[:, -1] > last_close).sum())
    if up >= DIRECTION_THRESHOLD * len(paths):
        return "Bullish", up
    if len(paths) - up >= DIRECTION_THRESHOLD * len(paths):
        return "Bearish", up
    return "No clear direction", up


def day_ticks(times: pd.DatetimeIndex, max_ticks: int = 9) -> tuple[list[int], list[str]]:
    """Positions and labels for the first bar of each day, thinned to at most `max_ticks`."""
    firsts = [i for i in range(len(times)) if i == 0 or times[i].date() != times[i - 1].date()][1:]
    step = max(1, -(-len(firsts) // max_ticks))
    firsts = firsts[::step]
    return firsts, [f"{times[i]:%b} {times[i].day}" for i in firsts]


def plot(candles: pd.DataFrame, bands: pd.DataFrame, verdict: str, runs_up: int,
         args: argparse.Namespace, device: str, path: Path) -> None:
    hist = candles["close"].tail(args.plot_history)
    last_close = hist.iloc[-1]

    # One x slot per candle, like a trading chart, so nights, weekends and session
    # breaks take no space. The forecast starts at the last close so it joins the history line.
    times = hist.index.append(pd.DatetimeIndex(bands.index))
    x_hist = np.arange(len(hist))
    x_fc = np.arange(len(hist) - 1, len(times))
    lo, med, hi = (np.r_[last_close, bands[c].to_numpy()] for c in ("p05", "median", "p95"))

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x_fc, lo, hi, color=FORECAST, alpha=0.18, linewidth=0, label="5–95% range")
    ax.plot(x_hist, hist.to_numpy(), color=HISTORY, linewidth=1.5, label="History (close)")
    ax.plot(x_fc, med, color=FORECAST, linewidth=2, label="Median path")
    ax.hlines(last_close, x_fc[0], x_fc[-1], color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
              label=f"Last close {last_close:,.2f}")
    ax.axvline(x_fc[0], color=GRID, linewidth=1, linestyle="--", zorder=0)

    # Direct labels on the forecast end values; nudge them apart if the band is narrow.
    y_min, y_max = ax.get_ylim()
    min_gap = 0.045 * (y_max - y_min)
    label_y = {"p95": max(hi[-1], med[-1] + min_gap), "median": med[-1], "p05": min(lo[-1], med[-1] - min_gap)}
    for key, value, color, weight in (("p95", hi[-1], TEXT_SECONDARY, "normal"),
                                      ("median", med[-1], TEXT_PRIMARY, "bold"),
                                      ("p05", lo[-1], TEXT_SECONDARY, "normal")):
        name = {"p95": "95%", "median": "median", "p05": "5%"}[key]
        ax.annotate(f"{name} {value:,.2f}", xy=(x_fc[-1], label_y[key]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=color, fontweight=weight, annotation_clip=False)

    # Verdict: icon + word + the counts behind it, so the call never rests on colour alone.
    icon, color = {"Bullish": ("▲", BULLISH), "Bearish": ("▼", BEARISH)}.get(verdict, ("◆", TEXT_SECONDARY))
    change = med[-1] / last_close - 1
    ax.text(1, 1.1, f"{icon} {verdict}", transform=ax.transAxes, ha="right", fontsize=14, fontweight="bold",
            color=color)
    ax.text(1, 1.025, f"{runs_up} of {args.runs} runs end higher · median {change:+.2%}", transform=ax.transAxes,
            ha="right", fontsize=9, color=TEXT_SECONDARY)

    ax.text(0, 1.1, f"{args.ticker} · next {args.pred_len}h with Kronos-small", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_title(f"{args.runs} runs · {args.lookback} hourly candles · {device} · "
                 f"last candle {hist.index[-1]:%Y-%m-%d %H:%M}", loc="left", fontsize=9, color=TEXT_SECONDARY)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"))
    ax.set_xticks(*day_ticks(times))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(0, len(times) - 1)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.07), ncol=4, frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.csv:
        print(f"Reading candles from {args.csv}...")
        candles = load_csv(args.csv, args.lookback)
    elif args.mt5:
        print(f"Reading H1 bars for {args.ticker} from MetaTrader 5...")
        candles = fetch_mt5(args.ticker, args.lookback)
    else:
        print(f"Downloading hourly candles for {args.ticker} from yfinance...")
        candles = fetch_yfinance(args.ticker, args.lookback)
    y_timestamp = future_timestamps(candles.index, args.pred_len)
    print(f"  {len(candles)} candles, {candles.index[0]} → {candles.index[-1]}, last close {candles['close'].iloc[-1]:,.2f}")

    print(f"Loading {MODEL_ID} on {device}...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    print(f"Sampling {args.runs} forecast paths of {args.pred_len}h...")
    paths = sample_paths(predictor, candles, y_timestamp, args)

    last_close = candles["close"].iloc[-1]
    p05, median, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    bands = pd.DataFrame({"p05": p05, "median": median, "p95": p95}, index=pd.DatetimeIndex(y_timestamp, name="timestamp"))
    verdict, runs_up = direction(paths, last_close)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9.-]+", "_", args.ticker)
    csv_path, png_path = args.out_dir / f"{stem}_forecast.csv", args.out_dir / f"{stem}_forecast.png"
    runs = pd.DataFrame(paths.T, index=bands.index, columns=[f"run_{i + 1:02d}" for i in range(args.runs)])
    pd.concat([bands, runs], axis=1).to_csv(csv_path, float_format="%.4f")
    plot(candles, bands, verdict, runs_up, args, device, png_path)

    print(f"\n{args.ticker} close in {args.pred_len}h (last close {last_close:,.2f}):")
    for name, value in (("5%", p05[-1]), ("median", median[-1]), ("95%", p95[-1])):
        print(f"  {name:>6}: {value:>12,.2f}  ({value / last_close - 1:+.2%})")
    print(f"\nDirection: {verdict} ({runs_up} of {args.runs} runs end above the last close; "
          f"a call needs {DIRECTION_THRESHOLD:.0%} agreement)")
    print(f"\nSaved {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
