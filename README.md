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

## Signal quality gates (why the bot may stay silent)
A model signal is only sent if it passes **both** layers:
1. **SOP checklist** (`signalbot/sop.py`): trend (EMA50/EMA200 + close vs EMA50), RSI not extreme (70/30),
   volatility within the 20–90th percentile of the last 500 bars, round-trip cost <= 0.15 R, and for forex an
   08:00–17:00 UTC entry session. The message shows each rule as OK/X.
2. **Strategy health gate** (`signalbot/health.py`): a fresh walk-forward backtest (after costs) must show
   >= 50 OOS trades, profit factor >= 1.2, positive expectancy, statistical significance (t >= 1.64), an edge over
   random sides, and positive expectancy recently (180 days; 365 for daily).
`python -m signalbot health EURUSD --interval 4h` shows the verdict; `--no-sop` / `--no-health-gate` disable a layer
(not recommended). **Current verdict: all four tested pairs (BTC, ETH, EURUSD, GBPUSD on 4h) FAIL the gate
(PF 0.70–0.79), so the bot sends nothing.** That is intended: no demonstrated edge, no signal.
No gate can guarantee profit; it only stops signals from a strategy that is not currently working.

## Telegram alerts
1. In Telegram, chat with **@BotFather** → `/newbot` → copy the **bot token**.
2. Open your new bot and press **Start** (for a group/channel: add the bot, post a message; a channel needs the bot as admin).
3. `export TELEGRAM_BOT_TOKEN=...` then `python -m signalbot chatid` → copy the `TELEGRAM_CHAT_ID`.
4. `export TELEGRAM_CHAT_ID=...` then `python -m signalbot telegram-test` (you should get a message).
5. Run the watcher (checks right after every bar close, sends only new signals, no repeats while a same-side trade is "open"):
```bash
python -m signalbot watch BTCUSDT ETHUSDT --interval 4h            # live, keeps running
python -m signalbot watch BTCUSDT --interval 4h --dry-run --once    # preview without sending
```
Keep it running 24/7 on a VPS / always-on machine (e.g. `tmux`, `systemd`, or Docker), or use `--once` from cron
(`5 */4 * * *`). Secrets stay in environment variables; never commit them. `data/` (cache + dedupe state) is git-ignored.

## Daily timeframe + funding rate (tested)
`--interval 1d` (crypto since 2019, forex since 2016) and `--funding` (Binance perpetual funding-rate features from
the public archive, crypto only). On 1d costs are only ~0.03-0.05 R, so results sit near break-even, but no variant
shows a statistically significant edge. Best-looking variant: ETHUSDT 1d + funding + SOP, 57 trades, PF 1.27,
+0.10 R/trade, but t = 0.83, 95% CI [-0.13, +0.32] R and -0.11 R over the last year: consistent with luck after trying
many variants. The health gate therefore also requires t >= 1.64 (one-sided 95%) and blocks all four pairs.

## Trend-following mode (rule-based, no ML)
`python -m signalbot trend BTCUSDT ETHUSDT --system 2` (1 = Turtle 20/10, 2 = 55/20; daily). Donchian breakout entry at
the next open, stop 2 ATR, exit on the opposite channel. No fitted parameters; both systems were fixed before testing.
Findings (2016/2019-2026): crypto longs are positive (pooled t about 1.8-1.9) but driven by 2020 and by crypto's bull
market (BTC x9, ETH x13 buy&hold), roughly flat over the last 2 years; crypto shorts and all forex are not profitable.
No single pair reaches t >= 1.64 on its own, so the per-symbol, per-direction health gate (>= 30 trades, PF >= 1.2,
expectancy > 0, t >= 1.64, positive last 2 years) blocks every signal by default. `--paper` still emits breakout signals,
clearly labelled PAPER / UNVALIDATED, so the idea can be forward-tested without risking money.
```bash
python -m signalbot trend BTCUSDT ETHUSDT --system 2 --paper --notify --loop   # checks after each daily close (00:00 UTC)
python -m signalbot trend BTCUSDT ETHUSDT --system 2 --paper --dry-run          # preview only
```
Each breakout bar is sent at most once. Use crypto only for paper trading; forex has no edge here.

## Automatic daily run (pick ONE)
**A) Windows Task Scheduler (free, your PC must be on at the scheduled time).** In PowerShell, from the project folder:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_task.ps1
Start-ScheduledTask -TaskName SignalbotTrendPaper      # test now
Get-Content data\task.log -Tail 30                      # see what happened
```
The script asks for the bot token (hidden) and Chat ID and stores them as user environment variables.
**B) GitHub Actions** (`.github/workflows/trend-signal.yml`, no PC needed): add repository secrets `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID`; it runs daily 00:20 UTC. Without the secrets it only does a dry run, so A and B never double-send
unless you set the secrets in both. Use "Run workflow" with `send_test` to verify the secrets.

## Scalping mode (15m, UNVALIDATED)
`python -m signalbot scalp BTCUSDT --once --dry-run` (drop `--dry-run` to send; without `--once` it loops every 15 min).
BTC 15m, ML model, horizon 16 bars (4 h), settings fixed before testing. Walk-forward since May 2024: ~7.5 signals/day,
expectancy +0.04 R before fees but -0.18 R at 8 bps and -0.50 R at 20 bps round trip (profit factor 0.69 / 0.32).
Break-even needs fees + spread + slippage below ~2 bps. Every message carries an UNVALIDATED label; `--max-per-day`
(default 6) caps messages per UTC day; `--sop` adds the SOP checklist (about 0.5 signals/day, still negative).
Windows: run `scripts\windows\setup_task.ps1` first, then `scripts\windows\setup_scalp_task.ps1` (every 15 minutes).
The scalp message is short: direction, Entry (reference price = last close), Julat masuk (entry zone = entry +/- 0.25 ATR; do not
chase outside it), TP, SL (both 1.5 ATR from entry), expiry (16 bars = 4 h), the UNVALIDATED label and the daily counter.
`--full-message` restores the detailed format with model probabilities.

## No pop-up windows
The scheduled tasks start through `wscript.exe` + `scripts\windows\run_hidden.vbs`, so no CMD window appears. If you set the
tasks up with an older version, run `powershell -ExecutionPolicy Bypass -File scripts\windows\hide_windows.ps1` once.

## Speed
Data refresh is incremental (only bars after the cached ones are downloaded; `fetch` does a full download).
`demo` takes ~1-2 s, `watch` waits 5 s after a bar closes for crypto (Binance candles are final immediately)
and 60 s for forex (Yahoo lags ~1 min). The health gate retrains walk-forward models and takes longer, but only runs
when a signal has already passed the SOP.

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

## Results (4h, same method, horizon 12 bars chosen before looking at results)
| Symbol | Trades | Win rate | Expectancy | Random-side baseline | Model edge |
|---|---|---|---|---|---|
| BTCUSDT | 876 | 47.0% | −0.14 R | −0.11 R | −0.03 R |
| ETHUSDT | 839 | 47.4% | −0.11 R | −0.08 R | −0.03 R |
| EURUSD | 549 | 50.3% | −0.07 R | −0.07 R | −0.00 R |
| GBPUSD | 551 | 47.0% | −0.10 R | −0.05 R | −0.05 R |

4h cuts the cost drag (fewer trades, wider stops) but the model still adds nothing over random sides,
so every pair still loses. Forex 4h bars are resampled from Yahoo 1h data (Yahoo has no native 4h).
Use `--interval 4h` on `fetch`, `backtest` and `signal`.

## Ideas that might help (test each with the same walk-forward harness)
- Daily timeframe (4h was tested, still no edge); lower-cost venues/pairs.
- Extra data: funding rate / open interest (crypto), economic calendar, news sentiment (LLM).
- Regime filters, meta-labelling, per-symbol models, longer history.
- Always paper trade for weeks before risking real money.

## Known simplifications
Entry at next open without slippage modelling beyond the flat cost; both-barriers-in-one-candle counted as a loss;
Yahoo hourly forex has no real volume; no position-size/leverage/margin limits.
