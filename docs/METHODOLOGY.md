# METHODOLOGY — trade-eda

Formulas, references, and honest limitations.

## Correlation

**Pearson** (product-moment): for aligned series x, y,

ρ = Σ(xᵢ−x̄)(yᵢ−ȳ) / √[Σ(xᵢ−x̄)² Σ(yᵢ−ȳ)²],  ddof = 1.

Measures *linear* co-movement. Sensitive to outliers — one flash-crash bar
shared by two names inflates ρ.

**Spearman** (rank): Pearson applied to average ranks, ties averaged.
Invariant to monotone transformations, so it captures monotonic
relationships in heavy-tailed return data where Pearson understates them.
Converges to Pearson under bivariate normality.

Both are computed on **log returns** by default (time-additive, symmetric
for gains/losses); simple returns optional.

## Covariance and Ledoit-Wolf shrinkage

The sample covariance S is unbiased but noisy: with n assets and t
observations its estimation error grows with n/t. Fed to a Markowitz
optimizer, the noise becomes extreme long/short weights (the optimizer
"discovers" spurious low-variance portfolios).

**Ledoit & Wolf (2004)**, "Honey, I Shrunk the Sample Covariance":

Σ* = δ·μI + (1−δ)·S,   μ = tr(S)/n,

with the *optimal* δ estimated from the data:

δ̂ = max(0, min(1, κ/γ)),  κ = max((π̂−ρ̂)/t, 0),

where π̂ sums the asymptotic variances of the entries of S, ρ̂ the
asymptotic covariance between S and the target, and γ = ‖S − μI‖²_F the
target's misspecification. Intuition: shrink hard when the sample matrix
is mostly noise (κ large relative to γ), trust it when it's precise.

The target μI (scaled identity) assumes average variance on the diagonal
and zero correlation — a deliberately dumb prior that is hard to beat
when data are scarce.

## Diversification

**Mean pairwise correlation** ρ̄: one number for "how correlated is this
book". **Effective number of bets** for an equal-weight portfolio:

N_eff = N / (1 + (N−1)·ρ̄).

Derivation: Var of an equal-weight portfolio with common variance σ² and
common correlation ρ̄ is σ²(1/N + (N−1)ρ̄/N); N_eff is the N that gives the
same variance at ρ̄ = 0. Eight names at ρ̄ = 0.6 ⇒ N_eff ≈ 2.3.

## Summary statistics

- Annualized vol: σ·√252 (daily bars).
- Skew/kurtosis: standardized central moments (excess kurtosis).
- **Jarque-Bera**: JB = n/6·(S² + K²/4) ~ χ²(2) under normality.
  JB > 9.21 rejects normality at 1%. Return series routinely fail — that's
  the point: it tells you which names break Gaussian risk models.

## Data quality

- *Missing closes*: None/NaN.
- *Stale-close runs*: ≥3 identical consecutive closes — a frozen feed or a
  halted name; correlations computed on stale prints are fiction.
- *Zero-volume bars*: prints with no trading behind them.
- *MAD outliers*: |x − median| > 8·MAD. MAD (median absolute deviation) is
  used instead of the standard deviation precisely because the standard
  deviation is inflated by the outliers being hunted.

## Limitations

- Daily-bar granularity; no intraday or microstructure-noise treatment.
- Correlation is linear (Pearson) or monotonic (Spearman) — nonlinear
  dependence (e.g., correlation that appears only in crashes) is not
  captured; that needs copulas or regime models.
- Ledoit-Wolf targets μI, not a factor structure; for strongly
  factor-driven universes a factor-model covariance can dominate it.
- Jarque-Bera's χ² approximation needs n ≳ 30; small samples under-reject.
- The demo universe is synthetic by design — it validates the machinery,
  not a market hypothesis.
