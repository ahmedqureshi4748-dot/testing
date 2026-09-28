# Kronos hourly forecast with yfinance

`forecast.py` downloads the last 400 completed hourly candles for a ticker with
yfinance and runs [Kronos](https://github.com/shiyu-coder/Kronos)-small 20
separate times (`sample_count=1` each), then plots the 5–95% range and the
per-hour median path, and calls a direction: **Bullish** when at least 60% of
runs end above the last close, **Bearish** when at least 60% end below,
otherwise *No clear direction*. `KronosPredictor` averages samples internally when
`sample_count > 1`, so independent runs are what give a real spread.

It runs on `cuda:0` when CUDA is available, otherwise on CPU (~2 s per run on 4 cores).

```bash
./setup.sh                                # clones Kronos, creates .venv (Python 3.12, uv if installed)
.venv/bin/python forecast.py BTC-USD      # -> output/BTC-USD_forecast.png + .csv
.venv/bin/python forecast.py AAPL --pred-len 14 --runs 50
```

### Gold (XAUUSD) from MetaTrader 5

Yahoo Finance has no spot XAUUSD (`GC=F` is COMEX gold futures), so read the
candles from MT5 instead:

```bash
# Windows, with the MT5 terminal open and logged in
.venv\Scripts\pip install MetaTrader5
.venv\Scripts\python forecast.py XAUUSD --mt5

# Any OS: export H1 bars from MT5 (View → Symbols → Bars → XAUUSD, H1 → Export Bars)
.venv/bin/python forecast.py XAUUSD --csv XAUUSD_H1.csv
```

`--mt5` uses the account the terminal is logged into, or `MT5_LOGIN`,
`MT5_PASSWORD` and `MT5_SERVER` if they are set. Use your broker's exact symbol
name (some call it `GOLD` or `XAUUSD.m`). Times stay in broker server time and
tick volume stands in for volume. `--csv` also accepts any file with a time
column and open/high/low/close columns.

Options: `--lookback` (400), `--pred-len` hours (24), `--runs` (20),
`--temperature` (1.0), `--top-p` (0.9), `--seed` (0; run *i* uses seed+*i*),
`--plot-history` (120), `--out-dir` (`output/`). The CSV holds the p05/median/p95
bands plus every run's close path.

## Every timeframe at once (mtf_forecast.py)

`mtf_forecast.py` forecasts the next 12 candles on 1d, 4h, 2h, 1h, 30m, 15m,
5m, 3m and 1m (20 runs each), prints a table of which way each timeframe
leans, and pools the runs into three verdicts: **short-term** (1m+3m+5m+15m),
**intraday** (30m–4h) and **overall**.

```bash
# Windows, MT5 terminal open and logged in (live bars, no API key needed)
.venv\Scripts\pip install MetaTrader5
.venv\Scripts\python mtf_forecast.py XAUUSD --mt5

# Any OS: a folder of MT5 exports named XAUUSD_M1.csv, XAUUSD_M3.csv, ..., XAUUSD_D1.csv
.venv/bin/python mtf_forecast.py XAUUSD --csv-dir exports/

# Fewer timeframes or a longer horizon
.venv/bin/python mtf_forecast.py XAUUSD --mt5 --timeframes H4,H1,M15 --pred-len 24
```

The table shows, per timeframe: last close, the move over the last 12 bars,
the next candle's median close, the median and 5–95% range 12 candles out,
how many runs end higher, and the call. It is saved as
`output/XAUUSD_mtf.csv`, and `output/XAUUSD_mtf.png` has one chart per
timeframe with the verdicts on top. On a 4-core CPU the full set takes about
3 minutes.

Needs network access to `huggingface.co` (model weights) and Yahoo Finance
(`*.finance.yahoo.com`, `fc.yahoo.com`, `guce.yahoo.com`).
