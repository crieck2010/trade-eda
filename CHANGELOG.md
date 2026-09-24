# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-24

### Added
- Pearson and Spearman correlation matrices (pure-Python, average-rank ties)
- Sample covariance and Ledoit-Wolf (2004) optimal shrinkage toward μI
- `portfolio_variance` / `portfolio_volatility` helpers (w′Σw)
- Per-asset summary stats: mean, annualized vol, skew, excess kurtosis,
  Jarque-Bera, min/max
- Data-quality checks: missing closes, zero-volume bars, stale-close runs,
  MAD-based price outliers
- `analyze()` — one-call correlation/EDA report with diversification stats
  (mean/max pairwise correlation, equal-weight effective N)
- Seeded two-block factor-model demo universe (known correlation structure)
- Adapters: `corr_for_optimize`, `corr_for_risk`, `to_agent_eda_report`,
  `eda_forecast_inputs` (all lazy, plain-data)
- CLI: `matrix`, `cov`, `describe`, `quality`, `report`, `risk`, `agent`,
  `license`, `update-check`; demo source or wide closes CSV
- README with "the maths" section, `docs/ARCHITECTURE.md`,
  `docs/METHODOLOGY.md`, end-to-end example
- 30 tests, all passing
