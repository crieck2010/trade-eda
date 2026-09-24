# trade-eda

Correlation, covariance, and exploratory data analysis for the trade-suite.
Stdlib-only — no numpy, pandas, or scipy.

## Install

```bash
pip install git+https://github.com/crieck2010/trade-eda.git
```

## Quick start

```python
from trade_eda import analyze, demo_bars

bars = demo_bars(n_days=250)          # seeded 8-asset universe, or your own bars
rep = analyze(bars)                  # full correlation/EDA report, plain data

print(rep["diversification"])
# {'mean_pairwise_corr': 0.32, 'max_pairwise_corr': {'a': 'BLK1_1', ...},
#  'effective_n_equal_weight': 3.1}   # 8 names, ~3 independent bets

from trade_eda.adapters import corr_for_optimize
cov = corr_for_optimize(bars)        # annualized covariance for trade-optimize
```

## CLI

```bash
trade-eda matrix   --source demo            # Pearson correlation matrix (JSON)
trade-eda matrix   --source demo --method spearman
trade-eda cov      --source demo            # Ledoit-Wolf shrunk covariance
trade-eda describe --source demo            # per-asset moments
trade-eda quality  --source demo            # missing/stale/outlier flags
trade-eda report   --source demo            # the full analyze() report
trade-eda risk     --source demo            # concentration report for trade-risk
trade-eda agent    --source demo            # compact brief for trade-agents
```

`--source` also accepts a wide closes CSV (`date,SYM1,SYM2,...`).

## The maths

**What you learn.** How correlation is measured (Pearson for linear co-movement, Spearman for monotonic relationships that survive outliers), why raw sample covariances mislead optimizers when assets outnumber observations, how shrinkage fixes that without a tuning parameter, and what "8 names" really means for diversification (usually ~3 independent bets). Plus the EDA hygiene that comes first: moments, normality checks, and the data-quality flags that catch frozen feeds before they corrupt a backtest.

**Why it matters.** Every downstream system in this suite consumes a covariance matrix: the optimizer turns it into weights, the risk engine turns it into limits, the Monte Carlo engine turns it into joint scenarios. A noisy covariance → extreme weights, false confidence, and VaR numbers that lie. Correlation is also the cheapest risk diagnostic there is — one matrix shows whether a "diversified" book is really one crowded trade.

**The maths.** Each implemented from scratch in this repo:

- *Pearson correlation:* ρ = Cov(x,y)/(σ_x·σ_y), in [−1, 1]. Constant series carry no linear signal → reported as 0, not NaN.
- *Spearman correlation:* Pearson applied to average ranks (ties averaged). Invariant to monotone transforms — catches relationships Pearson misses when returns are heavy-tailed.
- *Ledoit-Wolf shrinkage* (2004): Σ* = δ·μI + (1−δ)·S, where S is the sample covariance, μI the scaled-identity target, and δ ∈ [0, 1] the *optimal* intensity estimated from the data (δ̂ = max(0, min(1, κ/γ)) with κ from the asymptotic variance of S). No tuning knob; δ → 1 exactly when the data are too noisy to trust.
- *Effective number of bets* (equal-weight): N_eff = N / (1 + (N−1)·ρ̄). Eight names at ρ̄ = 0.6 behave like ~2.3 independent bets.
- *Jarque-Bera:* JB = n/6·(skew² + kurt²/4) ~ χ²(2) under normality; flags the non-normal return series that break Gaussian risk models.
- *MAD outliers:* |x − median| > 8·MAD — robust to the very outliers it hunts, unlike z-scores.

Full derivations and limitations: `docs/METHODOLOGY.md`.

## Interop

| Sibling | Adapter |
|---|---|
| trade-optimize | `corr_for_optimize` — shrunk annualized covariance → optimizer input |
| trade-risk | `corr_for_risk` — concentration report, most-correlated pair, effective N |
| trade-agents | `to_agent_eda_report` — compact JSON EDA brief for researchers |
| trade-volforecast | `eda_forecast_inputs` — vols + correlation for portfolio vol via w′Σw |

All sibling imports are lazy — trade-eda imports and tests clean with nothing else installed.

## Docs

- `docs/ARCHITECTURE.md` — module map and design decisions
- `docs/METHODOLOGY.md` — estimators, formulas, and honest limitations
- `examples/eda_example.py` — end-to-end walkthrough

## License

MIT. See `LICENSE`.
