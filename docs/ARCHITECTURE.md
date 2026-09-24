# ARCHITECTURE — trade-eda

## Purpose

The suite's correlation/covariance/EDA engine: one place that turns
`{symbol: bars}` into a correlation matrix, a well-conditioned covariance
matrix, per-asset summary statistics, and data-quality flags — all as plain
JSON-serializable data.

## Module map

| Module | Responsibility |
|---|---|
| `correlation.py` | `aligned_returns` (common-length truncation, log/simple), Pearson, Spearman (average-rank ties), full matrix, mean/max pairwise helpers |
| `covariance.py` | `sample_cov`, `ledoit_wolf` (2004, toward μI), `portfolio_variance`/`portfolio_volatility` |
| `describe.py` | `describe_series`/`describe_returns` (moments, Jarque-Bera), `data_quality` (missing/stale/zero-volume/MAD outliers) |
| `demo.py` | seeded two-block factor-model universe → known correlation structure for demos/tests |
| `__init__.py` | `analyze()` — orchestrates the full report; the function dashboards and the suite workflow call |
| `adapters.py` | lazy, plain-data bridges to trade-optimize / trade-risk / trade-agents / trade-volforecast |
| `cli.py` | `trade-eda` command: matrix, cov, describe, quality, report, risk, agent, license, update-check |
| `licensing.py` | license-key + update-check hooks (suite-wide convention, stubbed) |

## Design decisions

- **Stdlib-only.** No numpy/pandas/scipy — same constraint as every suite
  engine, so the package installs anywhere including the desktop .exe build.
- **Plain data across every boundary.** Functions take dicts/lists of floats
  and return dicts/lists. No engine objects leak into dashboards, agents, or
  sibling engines; JSON round-trips without a serializer.
- **Truncate, don't impute.** Ragged series are cut to their common length
  (oldest-first). Correlation on imputed data invents relationships; the
  engine refuses to.
- **Constant series → 0, not NaN.** A flat price series carries no linear
  signal; returning 0 keeps matrices finite and dashboards rendering.
- **Ledoit-Wolf default.** Optimizers and risk models get the shrunk
  covariance unless they explicitly ask for the sample version — a noisy
  covariance is the more dangerous default.
- **Lazy adapters.** `adapters.py` imports sibling engines inside functions,
  so trade-eda has zero install-time coupling to the rest of the suite.
- **Seeded demo with known structure.** Two correlation blocks (~0.6
  within, ~0.1 across) make demos self-checking: if the heatmap doesn't
  show blocks, something broke.

## Scaling notes

- All hot loops are O(n²·t) or better for n assets, t observations — fine
  into the hundreds of assets on daily bars. Ledoit-Wolf's π̂ term is the
  heaviest (n²·t); for very wide universes (n > 500) prefer
  `shrinkage="sample"` or pre-aggregate.
- `analyze()` is stateless and pure: reports for disjoint universes can be
  computed in parallel processes and concatenated; no shared state.

## Interop contract

- Input: `{SYMBOL: [bar]}` where a bar is a dict (or object) with `close`;
  `volume`/`timestamp` optional (used by data-quality when present).
- Output: nested plain dicts/lists/numbers/strings — JSON-safe.
- Errors: `ValueError` for bad input (too few symbols/observations, unknown
  method), `RuntimeError` from adapters when a sibling engine is missing
  (message names the `pip install` fix).
