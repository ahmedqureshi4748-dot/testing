#!/usr/bin/env python3
"""Multi-timeframe Kronos-small forecast: which way is each timeframe leaning?

For each timeframe from 1d down to 1m this reads the last 400 closed candles,
runs Kronos-small 20 separate times (sample_count=1) for the next N candles,
and calls the direction from how many runs end above the last close. The
1m/3m/5m/15m runs are pooled into one short-term verdict, 30m-4h into an
intraday verdict, and every timeframe into an overall bias.

Usage:
    python mtf_forecast.py XAUUSD --mt5                  # live bars from a running MT5 terminal (Windows)
    python mtf_forecast.py XAUUSD --csv-dir exports/     # MT5 exports named XAUUSD_M15.csv, XAUUSD_H4.csv, ...
    python mtf_forecast.py XAUUSD --mt5 --timeframes H1,M15,M5 --pred-len 24
"""

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import forecast as fc  # also puts Kronos on sys.path and selects the Agg backend
from model import Kronos, KronosPredictor, KronosTokenizer

# MT5 timeframe name -> (label, bar length), longest first.
TIMEFRAMES = {
    "D1": ("1d", pd.Timedelta(days=1)),
    "H4": ("4h", pd.Timedelta(hours=4)),
    "H2": ("2h", pd.Timedelta(hours=2)),
    "H1": ("1h", pd.Timedelta(hours=1)),
    "M30": ("30m", pd.Timedelta(minutes=30)),
    "M15": ("15m", pd.Timedelta(minutes=15)),
    "M5": ("5m", pd.Timedelta(minutes=5)),
    "M3": ("3m", pd.Timedelta(minutes=3)),
    "M1": ("1m", pd.Timedelta(minutes=1)),
}
GROUPS = {
    "Short-term (1m+3m+5m+15m)": ["M1", "M3", "M5", "M15"],
    "Intraday (30m–4h)": ["M30", "H1", "H2", "H4"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("symbol", nargs="?", default="XAUUSD", help="MT5 symbol (default: XAUUSD)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mt5", action="store_true", help="read live bars from a running MetaTrader 5 terminal (Windows)")
    src.add_argument("--csv-dir", type=Path, help="folder of MT5 exports named SYMBOL_TF.csv (e.g. XAUUSD_M15.csv)")
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES),
                   help=f"comma-separated MT5 timeframes (default: {','.join(TIMEFRAMES)})")
    p.add_argument("--lookback", type=int, default=400, help="candles fed to the model per timeframe (default: 400)")
    p.add_argument("--pred-len", type=int, default=12, help="candles to forecast on every timeframe (default: 12)")
    p.add_argument("--runs", type=int, default=20, help="independent predictor runs per timeframe (default: 20)")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature T (default: 1.0)")
    p.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling top_p (default: 0.9)")
    p.add_argument("--seed", type=int, default=0, help="run i is seeded with seed+i (default: 0)")
    p.add_argument("--plot-history", type=int, default=60, help="history candles per chart panel (default: 60)")
    p.add_argument("--out-dir", type=Path, default=Path("output"), help="where the PNG and CSV go (default: output/)")
    args = p.parse_args()
    args.timeframes = [tf.strip().upper() for tf in args.timeframes.split(",") if tf.strip()]
    unknown = [tf for tf in args.timeframes if tf not in TIMEFRAMES]
    if unknown:
        p.error(f"unknown timeframe(s) {', '.join(unknown)}; choose from {', '.join(TIMEFRAMES)}")
    return args


def csv_for(folder: Path, symbol: str, tf: str) -> Path | None:
    """MT5 export for one timeframe: SYMBOL_TF.csv, or SYMBOL_TF_<dates>.csv as MT5 names it."""
    for pattern in (f"{symbol}_{tf}.csv", f"{symbol}_{tf}_*.csv"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[-1]
    return None


def load_all(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Candles for every requested timeframe, all read up front so they share one moment in time."""
    candles = {}
    for tf in args.timeframes:
        if args.mt5:
            candles[tf] = fc.fetch_mt5(args.symbol, args.lookback, tf)
        else:
            path = csv_for(args.csv_dir, args.symbol, tf)
            if path is None:
                print(f"  {TIMEFRAMES[tf][0]:>4}: skipped, no {args.symbol}_{tf}.csv in {args.csv_dir}")
                continue
            candles[tf] = fc.load_csv(path, args.lookback)
        print(f"  {TIMEFRAMES[tf][0]:>4}: {len(candles[tf])} candles up to {candles[tf].index[-1]}")
    if not candles:
        sys.exit("no candles loaded")
    return candles


def forecast_timeframe(predictor: KronosPredictor, tf: str, candles: pd.DataFrame,
                       args: argparse.Namespace) -> dict:
    label, step = TIMEFRAMES[tf]
    y_timestamp = fc.future_timestamps(candles.index, args.pred_len, step)
    paths = fc.sample_paths(predictor, candles, y_timestamp, args, verbose=False)

    close = candles["close"]
    last_close = close.iloc[-1]
    p05, median, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    verdict, runs_up = fc.direction(paths, last_close)
    return {
        "tf": tf, "label": label, "step": step, "last_time": close.index[-1], "last_close": last_close,
        "recent_change": last_close / close.iloc[-1 - args.pred_len] - 1,
        "next_candle": median[0], "end_median": median[-1], "end_p05": p05[-1], "end_p95": p95[-1],
        "end_change": median[-1] / last_close - 1, "runs_up": runs_up, "runs": len(paths), "verdict": verdict,
        "history": close.tail(args.plot_history),
        "bands": pd.DataFrame({"p05": p05, "median": median, "p95": p95}, index=pd.DatetimeIndex(y_timestamp)),
    }


def pooled_verdict(rows: list[dict]) -> tuple[str, int, int]:
    """Direction from all runs of several timeframes pooled together."""
    up, total = sum(r["runs_up"] for r in rows), sum(r["runs"] for r in rows)
    if up >= fc.DIRECTION_THRESHOLD * total:
        return "Bullish", up, total
    if total - up >= fc.DIRECTION_THRESHOLD * total:
        return "Bearish", up, total
    return "No clear direction", up, total


def icon(verdict: str) -> str:
    return {"Bullish": "▲", "Bearish": "▼"}.get(verdict, "◆")


def verdict_color(verdict: str) -> str:
    return {"Bullish": fc.BULLISH, "Bearish": fc.BEARISH}.get(verdict, fc.TEXT_SECONDARY)


def print_table(rows: list[dict], summary: list[tuple[str, str, int, int]], args: argparse.Namespace) -> None:
    n = args.pred_len
    header = (f"{'TF':<4}  {'Last close':>11}  {f'Last {n} bars':>12}  {'Next candle':>13}  "
              f"{f'In {n} candles':>20}  {'5–95% range':>21}  {'Runs up':>7}  Direction")
    print("\n" + header + "\n" + "─" * len(header))
    for r in rows:
        next_arrow = "▲" if r["next_candle"] > r["last_close"] else "▼"
        print(f"{r['label']:<4}  {r['last_close']:>11,.2f}  {r['recent_change']:>+12.2%}  "
              f"{r['next_candle']:>11,.2f} {next_arrow}  "
              f"{r['end_median']:>11,.2f} ({r['end_change']:+.2%})  "
              f"{r['end_p05']:>10,.2f}–{r['end_p95']:<10,.2f}  "
              f"{r['runs_up']:>3}/{r['runs']:<3}  {icon(r['verdict'])} {r['verdict']}")
    print()
    for name, verdict, up, total in summary:
        print(f"{name:<27} {icon(verdict)} {verdict} ({up} of {total} runs end higher)")


def save_csv(rows: list[dict], path: Path) -> None:
    table = pd.DataFrame([{
        "timeframe": r["label"], "last_candle": r["last_time"], "last_close": r["last_close"],
        "recent_change_pct": 100 * r["recent_change"], "next_candle_median": r["next_candle"],
        "end_median": r["end_median"], "end_change_pct": 100 * r["end_change"], "end_p05": r["end_p05"],
        "end_p95": r["end_p95"], "runs_up": r["runs_up"], "runs": r["runs"], "direction": r["verdict"],
    } for r in rows])
    table.to_csv(path, index=False, float_format="%.4f")


def plot_dashboard(rows: list[dict], summary: list[tuple[str, str, int, int]], args: argparse.Namespace,
                   device: str, path: Path) -> None:
    cols = 3
    nrows = -(-len(rows) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(15, 3.3 * nrows + 2.2), dpi=130, facecolor=fc.SURFACE,
                             squeeze=False)

    for ax, r in zip(axes.flat, rows):
        hist, bands = r["history"], r["bands"]
        last_close = r["last_close"]
        x_hist = np.arange(len(hist))
        x_fc = np.arange(len(hist) - 1, len(hist) + len(bands))
        lo, med, hi = (np.r_[last_close, bands[c].to_numpy()] for c in ("p05", "median", "p95"))

        ax.set_facecolor(fc.SURFACE)
        ax.fill_between(x_fc, lo, hi, color=fc.FORECAST, alpha=0.18, linewidth=0)
        ax.plot(x_hist, hist.to_numpy(), color=fc.HISTORY, linewidth=1.2)
        ax.plot(x_fc, med, color=fc.FORECAST, linewidth=1.8)
        ax.hlines(last_close, x_fc[0], x_fc[-1], color=fc.TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.axvline(x_fc[0], color=fc.GRID, linewidth=0.8, linestyle="--", zorder=0)

        ax.set_title(r["label"], loc="left", fontsize=13, fontweight="bold", color=fc.TEXT_PRIMARY)
        ax.set_title(f"{icon(r['verdict'])} {r['verdict']}  {r['runs_up']}/{r['runs']} · {r['end_change']:+.2%}",
                     loc="right", fontsize=10, fontweight="bold", color=verdict_color(r["verdict"]))

        times = hist.index.append(bands.index)
        fmt = "%b %d" if r["step"] >= pd.Timedelta(days=1) else ("%b %d %H:%M" if r["step"] >= pd.Timedelta(hours=1)
                                                                 else "%H:%M")
        ticks = [0, len(hist) - 1]
        ax.set_xticks(ticks, [f"{times[i]:{fmt}}" for i in ticks])
        ax.set_xlim(0, len(times) - 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"))
        ax.grid(axis="y", color=fc.GRID, linewidth=0.7)
        ax.tick_params(colors=fc.TEXT_SECONDARY, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(fc.GRID)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)

    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.3 / height, f"{args.symbol} · every timeframe, next {args.pred_len} candles",
             fontsize=17, fontweight="bold", color=fc.TEXT_PRIMARY, va="top")
    fig.text(0.01, 1 - 0.68 / height, f"Kronos-small · {args.runs} runs per timeframe · {args.lookback} candles each · "
             f"{device} · last candle {max(r['last_time'] for r in rows):%Y-%m-%d %H:%M}",
             fontsize=9.5, color=fc.TEXT_SECONDARY, va="top")
    for i, (name, verdict, up, total) in enumerate(summary):
        x = 0.01 + i * 0.33
        fig.text(x, 1 - 1.02 / height, name, fontsize=9.5, color=fc.TEXT_SECONDARY, va="top")
        fig.text(x, 1 - 1.24 / height, f"{icon(verdict)} {verdict} · {up}/{total} runs up", fontsize=13,
                 fontweight="bold", color=verdict_color(verdict), va="top")
    fig.text(0.01, 0.12 / height, "Grey: history · Blue: median path with 5–95% range · Dashed: last close · "
             f"{icon('Bullish')}/{icon('Bearish')} needs {fc.DIRECTION_THRESHOLD:.0%} of runs to agree",
             fontsize=9, color=fc.TEXT_SECONDARY, va="bottom")

    fig.tight_layout(rect=(0, 0.4 / height, 1, 1 - 1.75 / height), h_pad=2.2, w_pad=2.5)
    fig.savefig(path, facecolor=fc.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Reading {args.symbol} candles from {'MetaTrader 5' if args.mt5 else args.csv_dir}...")
    candles = load_all(args)

    print(f"Loading {fc.MODEL_ID} on {device}...")
    predictor = KronosPredictor(Kronos.from_pretrained(fc.MODEL_ID), KronosTokenizer.from_pretrained(fc.TOKENIZER_ID),
                                device=device, max_context=512)

    print(f"Forecasting the next {args.pred_len} candles, {args.runs} runs per timeframe...")
    rows = []
    for tf, df in candles.items():
        start = time.perf_counter()
        rows.append(forecast_timeframe(predictor, tf, df, args))
        r = rows[-1]
        print(f"  {r['label']:>4}: {icon(r['verdict'])} {r['verdict']} ({r['runs_up']}/{r['runs']} runs up) "
              f"in {time.perf_counter() - start:.0f}s")

    summary = [(name, *pooled_verdict(group)) for name, tfs in GROUPS.items()
               if (group := [r for r in rows if r["tf"] in tfs])]
    summary.append((f"Overall ({len(rows)} timeframes)", *pooled_verdict(rows)))
    print_table(rows, summary, args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9.-]+", "_", args.symbol)
    csv_path, png_path = args.out_dir / f"{stem}_mtf.csv", args.out_dir / f"{stem}_mtf.png"
    save_csv(rows, csv_path)
    plot_dashboard(rows, summary, args, device, png_path)
    print(f"\nSaved {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
