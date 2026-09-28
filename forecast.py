#!/usr/bin/env python3
"""Probabilistic hourly forecast for a yfinance ticker with Kronos-small.

KronosPredictor averages its samples internally when sample_count > 1, which
collapses the distribution into a single path. To get an uncertainty band we
instead run the predictor N separate times with sample_count=1 and take
per-step quantiles over the independent paths.

Usage:
    python forecast.py BTC-USD
    python forecast.py AAPL --pred-len 14 --runs 50

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

import matplotlib.dates as mdates
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

# Plot palette: forecast in the primary series blue, history as recessive neutral ink.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
HISTORY = "#8a8984"
FORECAST = "#2a78d6"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ticker", nargs="?", default="BTC-USD", help="yfinance ticker (default: BTC-USD)")
    p.add_argument("--lookback", type=int, default=400, help="hourly candles fed to the model (default: 400)")
    p.add_argument("--pred-len", type=int, default=24, help="hours to forecast (default: 24)")
    p.add_argument("--runs", type=int, default=20, help="independent predictor runs (default: 20)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature T (default: 1.0)")
    p.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling top_p (default: 0.9)")
    p.add_argument("--seed", type=int, default=0, help="run i is seeded with seed+i (default: 0)")
    p.add_argument("--plot-history", type=int, default=120, help="history candles shown in the plot (default: 120)")
    p.add_argument("--out-dir", type=Path, default=Path("output"), help="where the PNG and CSV go (default: output/)")
    return p.parse_args()


def fetch_candles(ticker: str, lookback: int) -> pd.DataFrame:
    """Last `lookback` completed hourly OHLCV candles, indexed by naive exchange-local time."""
    df = yf.Ticker(ticker).history(period="1y", interval="1h")
    if df.empty:
        sys.exit(f"yfinance returned no hourly data for {ticker!r}")

    # Drop the still-forming candle so the model only sees closed bars.
    if df.index[-1] + pd.Timedelta(hours=1) > pd.Timestamp.now(tz=df.index.tz):
        df = df.iloc[:-1]

    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna().astype("float64")
    df.index = df.index.tz_localize(None)
    if len(df) < lookback:
        sys.exit(f"only {len(df)} hourly candles available for {ticker!r}, need {lookback}")
    return df.tail(lookback)


def future_timestamps(index: pd.DatetimeIndex, n: int) -> pd.Series:
    """Next `n` hourly bar times, restricted to the (weekday, hour, minute) slots seen in history.

    That gives every hour for 24/7 markets like crypto, and only session hours
    on trading days for equities.
    """
    slots = set(zip(index.dayofweek, index.hour, index.minute))
    out, t = [], index[-1]
    while len(out) < n:
        t += pd.Timedelta(hours=1)
        if (t.dayofweek, t.hour, t.minute) in slots:
            out.append(t)
    return pd.Series(out)


def sample_paths(predictor: KronosPredictor, candles: pd.DataFrame, y_timestamp: pd.Series,
                 args: argparse.Namespace) -> np.ndarray:
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
        print(f"  run {i + 1:>2}/{args.runs}: close in {args.pred_len}h = {paths[-1][-1]:,.2f} "
              f"({time.perf_counter() - start:.1f}s)")
    return np.vstack(paths)


def plot(candles: pd.DataFrame, bands: pd.DataFrame, args: argparse.Namespace, device: str, path: Path) -> None:
    hist = candles["close"].tail(args.plot_history)
    last_t, last_close = hist.index[-1], hist.iloc[-1]

    # Start the forecast at the last observed close so it joins the history line.
    t = pd.DatetimeIndex([last_t]).append(pd.DatetimeIndex(bands.index))
    lo, med, hi = (np.r_[last_close, bands[c].to_numpy()] for c in ("p05", "median", "p95"))

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(t, lo, hi, color=FORECAST, alpha=0.18, linewidth=0, label="5–95% range")
    ax.plot(hist.index, hist.to_numpy(), color=HISTORY, linewidth=1.5, label="History (close)")
    ax.plot(t, med, color=FORECAST, linewidth=2, label="Median path")
    ax.axvline(last_t, color=GRID, linewidth=1, linestyle="--", zorder=0)

    # Direct labels on the forecast end values; nudge them apart if the band is narrow.
    y_min, y_max = ax.get_ylim()
    min_gap = 0.045 * (y_max - y_min)
    label_y = {"p95": max(hi[-1], med[-1] + min_gap), "median": med[-1], "p05": min(lo[-1], med[-1] - min_gap)}
    for key, value, color, weight in (("p95", hi[-1], TEXT_SECONDARY, "normal"),
                                      ("median", med[-1], TEXT_PRIMARY, "bold"),
                                      ("p05", lo[-1], TEXT_SECONDARY, "normal")):
        name = {"p95": "95%", "median": "median", "p05": "5%"}[key]
        ax.annotate(f"{name} {value:,.2f}", xy=(t[-1], label_y[key]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9, color=color, fontweight=weight, annotation_clip=False)

    ax.text(0, 1.1, f"{args.ticker} · next {args.pred_len}h with Kronos-small", transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_title(f"{args.runs} independent runs (sample_count=1) on the last {args.lookback} hourly candles · "
                 f"{device} · last candle {last_t:%Y-%m-%d %H:%M}", loc="left", fontsize=9, color=TEXT_SECONDARY)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"))
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(hist.index[0], t[-1])
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Downloading hourly candles for {args.ticker}...")
    candles = fetch_candles(args.ticker, args.lookback)
    y_timestamp = future_timestamps(candles.index, args.pred_len)
    print(f"  {len(candles)} candles, {candles.index[0]} → {candles.index[-1]}, last close {candles['close'].iloc[-1]:,.2f}")

    print(f"Loading {MODEL_ID} on {device}...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
    model = Kronos.from_pretrained(MODEL_ID)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    print(f"Sampling {args.runs} forecast paths of {args.pred_len}h...")
    paths = sample_paths(predictor, candles, y_timestamp, args)

    p05, median, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    bands = pd.DataFrame({"p05": p05, "median": median, "p95": p95}, index=pd.DatetimeIndex(y_timestamp, name="timestamp"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9.-]+", "_", args.ticker)
    csv_path, png_path = args.out_dir / f"{stem}_forecast.csv", args.out_dir / f"{stem}_forecast.png"
    runs = pd.DataFrame(paths.T, index=bands.index, columns=[f"run_{i + 1:02d}" for i in range(args.runs)])
    pd.concat([bands, runs], axis=1).to_csv(csv_path, float_format="%.4f")
    plot(candles, bands, args, device, png_path)

    last_close = candles["close"].iloc[-1]
    print(f"\n{args.ticker} close in {args.pred_len}h (last close {last_close:,.2f}):")
    for name, value in (("5%", p05[-1]), ("median", median[-1]), ("95%", p95[-1])):
        print(f"  {name:>6}: {value:>12,.2f}  ({value / last_close - 1:+.2%})")
    print(f"\nSaved {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
