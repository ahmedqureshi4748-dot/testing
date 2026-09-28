# Kronos hourly forecast with yfinance

`forecast.py` downloads the last 400 completed hourly candles for a ticker with
yfinance and runs [Kronos](https://github.com/shiyu-coder/Kronos)-small 20
separate times (`sample_count=1` each), then plots the 5–95% range and the
per-hour median path. `KronosPredictor` averages samples internally when
`sample_count > 1`, so independent runs are what give a real spread.

It runs on `cuda:0` when CUDA is available, otherwise on CPU (~2 s per run on 4 cores).

```bash
./setup.sh                                # clones Kronos, creates .venv (Python 3.12, uv if installed)
.venv/bin/python forecast.py BTC-USD      # -> output/BTC-USD_forecast.png + .csv
.venv/bin/python forecast.py AAPL --pred-len 14 --runs 50
```

Options: `--lookback` (400), `--pred-len` hours (24), `--runs` (20),
`--temperature` (1.0), `--top-p` (0.9), `--seed` (0; run *i* uses seed+*i*),
`--plot-history` (120), `--out-dir` (`output/`). The CSV holds the p05/median/p95
bands plus every run's close path.

Needs network access to `huggingface.co` (model weights) and Yahoo Finance
(`*.finance.yahoo.com`, `fc.yahoo.com`, `guce.yahoo.com`).
