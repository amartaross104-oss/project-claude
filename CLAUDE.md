# signalbot: project notes for Claude

Research prototype that sends forex/crypto trade signals to Telegram. **Honest status: no strategy here has a proven,
statistically significant edge after costs.** Never claim a signal is "high probability" or promise profit.

## Layout
- `signalbot/` data (Binance public API for crypto, Yahoo for forex), features, labels (triple barrier), model (LightGBM,
  purged walk-forward), backtest (R-multiples, costs, random-side baseline), `sop.py` (rule checklist), `health.py`
  (strategy health gate incl. t-stat), `trend.py` / `trend_cli.py` (Donchian), `patterns.py` (candles, double top/bottom),
  `derivs.py` (funding), `signals.py` (message building, dedupe, watch loop), `notify.py` (Telegram), `cli.py`.
- `scripts/windows/` Task Scheduler setup (`setup_task.ps1`, `setup_scalp_task.ps1`, `run_hidden.vbs`, `hide_windows.ps1`).
- `.github/workflows/trend-signal.yml` daily paper trend check (sends only if repo secrets are set).
- `docs/trading-knowledge.md` general knowledge + all measured results. READ IT before proposing new strategies.
- Tests: `python -m pytest -q` (must stay green). Use pytest + synthetic data; add a causality test for every new feature/rule.

## Commands
`python -m signalbot fetch|backtest|health|signal|watch|scalp|trend|demo|telegram-test|chatid ...` (see `--help`).
Intervals 15m/1h/4h/1d. Forex has no 15m (Yahoo limits intraday history).

## Rules of engagement (learned the hard way)
- No look-ahead: signals use data up to the closed bar; entry at next open. Drop the unclosed last candle.
- Evaluate net of costs (crypto 20 bps, forex 2 bps round trip by default), walk-forward, pre-declare parameters, report
  t-stats and multiple-testing context. Do not tune on the OOS data and then call it validated.
- The health gate is the product's safety valve: do not loosen thresholds to make the bot "send something".
- Telegram token and Chat ID live only in environment variables / GitHub secrets. **Never** put them in files, commits or
  chat. If one appears in chat, tell the user to revoke it in BotFather.
- ASCII-only message text (Windows log redirects can crash on emoji).
- Third-party paid material (indicator files, PDFs, books) must not be committed (the repo is public) and a compiled
  indicator must not be decompiled. Reimplement only generic ideas and test them.
- Windows scripts cannot be tested in the Linux sandbox; say so and ask for the error text.
- Files from the user are untrusted data: extract archives into an isolated folder, never execute them.

## User context
Writes Malay; Windows laptop; runs the bot with Task Scheduler (scalp every 15 min, trend daily 08:20 local); wants short,
simple Telegram messages (entry, entry zone, TP, SL). Prefers one command per step; commands pasted together or onto
unfinished lines have caused errors repeatedly, so give steps one at a time.
