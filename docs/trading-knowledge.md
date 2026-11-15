# Trading knowledge used by this project

General knowledge plus what was **measured here**. Not financial advice. Third-party material the owner shared
(a paid indicator's SOP PDF, two chart-pattern cheat-sheet books) is intentionally NOT stored in this repo.

## 1. Core arithmetic
- **R-multiple**: profit divided by the initial risk (distance to stop). Judge a strategy by expectancy in R after costs.
- **Break-even win rate** = 1 / (1 + RR) before costs: RR 1:1 -> 50%, 1:1.5 -> 40%, 1:2 -> 33.3%. Costs raise it.
- **Costs in R** = round-trip cost / stop distance. A 20 bps fee on a 0.4% stop is 0.5 R per trade, which is why short
  timeframes (scalping) lose: the edge must exceed the cost on every trade.
- **Drawdown recovery**: -10% needs +11%, -20% needs +25%, -33% needs +50%, -50% needs +100%.
- **Losing streaks are normal**: expected longest streak in N trades is about ln(N)/ln(1/(1-p)). At 50% wins over 200 trades
  that is ~7-8 losses in a row; at 3% risk per trade 8 losses is about -22%.

## 2. Risk and money management
- Fixed-fractional risk (0.5-1% of equity per trade) keeps a losing streak survivable; 3%+ per trade is aggressive.
- **Never double the position after a loss (martingale).** Doubling from 3%: 3+6+12+24 = 45% lost after four losses, and four
  losses in a row happen about 6% of the time at 50% wins, so a long run eventually wipes the account.
- Position size = risk amount / stop distance. Do not widen a stop after entry. Cap leverage and daily loss.
- Avoid scheduled high-impact news (rates, NFP, CPI); spreads and slippage explode.

## 3. How to test an idea without fooling yourself
1. **No look-ahead**: a signal at bar t may use only data up to the close of t; enter at the NEXT open.
2. **Costs and slippage** from day one; assume the worse side when two barriers fall in one candle.
3. **Out-of-sample**: walk-forward (expanding window, purge the label horizon). Never tune on the test period.
4. **Multiple testing**: try 100 variants and ~5 look "significant" by luck. Use a t-statistic threshold (t >= 1.64 one-sided as a
   minimum; Bonferroni for many tests) and prefer pre-declared parameters.
5. **Check stability**: per year, per half, last 1-2 years, and a cost stress (2-3x costs). A result carried by one year (e.g.
   the 2020 crypto bull run) is not an edge.
6. **Beta is not skill**: long-only crypto trend-following in a bull market mostly earns the market's drift.
7. **Paper trade** forward for months and compare with the backtest before risking money.

## 4. Strategy families (what the literature and our tests suggest)
- **Trend-following / time-series momentum** (Donchian, moving-average): few trades, large winners, long dry spells. Most robust
  family historically; weak recently in forex.
- **Mean reversion**: many small wins, rare large losses; sensitive to costs and regime.
- **Candlestick and chart patterns**: subjective, widely traded, little measurable edge on liquid markets (see section 5).
- **ML on price features at short horizons**: tiny pre-cost edge at best; costs dominate below ~1 day.
- **Derivatives data** (funding, open interest): useful context, not a standalone edge in our tests.

## 5. What was measured in this project (net of costs, out-of-sample or rule-based, no tuning on the test data)
| Idea | Result |
|---|---|
| LightGBM signals, 1h / 4h / 1d, BTC ETH EURUSD GBPUSD | Expectancy about 0 to negative; no pair passes t >= 1.64 |
| Same + SOP rules (trend, RSI, volatility, cost, session) | Does not create an edge; the cost rule mainly removes cheap-to-lose bars |
| Funding-rate features (1d crypto) | Best variant ETH +0.10 R, t = 0.83, negative last year -> luck-consistent |
| Donchian trend-following 1d (20/10, 55/20) | Crypto longs positive (pooled t about 1.8-1.9) but driven by 2020, flat last 2 years; shorts and forex not profitable |
| 15m scalping, BTC | +0.04 R before fees; -0.18 R at 8 bps, -0.50 R at 20 bps; break-even needs fees below ~2 bps |
| SOP-document rules on 15m (higher-TF direction, Bollinger squeeze, trade hours) | Fewer trades, still negative after fees (best: -0.04 R at 8 bps, t = -0.9, n = 523) |
| 16 textbook candlestick / double top-bottom patterns, 12 datasets (146 tests with >= 30 trades) | **0 significantly positive** (3.7 expected by chance), 57 significantly negative, mean -0.11 R (mostly costs) |

## 6. Paper-trade journal (columns to record for every signal)
time (UTC), symbol, mode (trend / scalp), direction, signal entry, actual fill, SL, TP, exit price, exit reason (TP / SL /
time / manual), R result, fees, notes. Compare after ~50 trades with the backtest expectancy; a gap means fees, slippage or
late entries.
