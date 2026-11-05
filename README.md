# signalbot — AI trade-signal prototype (forex + crypto)

Research prototype. **Not financial advice. It does not currently have a proven edge (see Results).**

## Pipeline
`data` (Binance public API for crypto, Yahoo Finance for forex) → `features` (RSI, EMA, MACD, ATR,
Bollinger, returns, time) → `labels` (triple-barrier: TP = SL = 1.5×ATR, max 24 bars, entry at next
open) → `model` (LightGBM, purged expanding-window walk-forward) → `backtest` (one position at a time,
costs included, results in R-multiples, compared with a random-side baseline) → `signal` (+ optional Telegram).

## Usage
```bash
pip install -r requirements.txt
python -m signalbot fetch BTCUSDT            # download + cache data (crypto: *USDT, forex: EURUSD etc.)
python -m signalbot backtest EURUSD          # walk-forward out-of-sample backtest
python -m signalbot signal ETHUSDT           # signal for the last closed bar
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... python -m signalbot signal BTCUSDT --notify
python -m pytest
```
Key flags: `--margin` (main trade filter: P(dir) − P(opposite)), `--thr`, `--tp-mult`, `--horizon`, `--cost-bps`.

## Results (1h, walk-forward OOS, default settings, costs: crypto 20 bps, forex 2 bps round trip)
| Symbol | Trades | Win rate | Expectancy | Random-side baseline |
|---|---|---|---|---|
| BTCUSDT | ~2800 | 49.5% | −0.23 R | −0.24 R |
| ETHUSDT | ~2860 | 51.3% | −0.13 R | −0.16 R |
| EURUSD | ~1900 | 49.7% | −0.13 R | −0.14 R |
| GBPUSD | ~1900 | 49.3% | −0.13 R | −0.13 R |

Honest reading: the model is only marginally better than random sides (BTC AUC ≈ 0.52, EURUSD ≈ 0.49),
and costs make every setup lose. Zero-cost expectancy is ≈ 0 R, which also confirms the accounting is correct.
A margin sweep (0.05–0.20) did not change that; further tuning on the same OOS data would just be overfitting.

## Ideas that might help (test each with the same walk-forward harness)
- Higher timeframe (4h/1d) so costs are a smaller fraction of the stop distance; lower-cost venues/pairs.
- Extra data: funding rate / open interest (crypto), economic calendar, news sentiment (LLM).
- Regime filters, meta-labelling, per-symbol models, longer history.
- Always paper trade for weeks before risking real money.

## Known simplifications
Entry at next open without slippage modelling beyond the flat cost; both-barriers-in-one-candle counted as a loss;
Yahoo hourly forex has no real volume; no position-size/leverage/margin limits.
