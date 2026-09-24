"""Tests for trade-eda (stdlib only, no network)."""

from __future__ import annotations

import math
import random

import pytest

import trade_eda
from trade_eda import correlation as C
from trade_eda import covariance as V
from trade_eda import describe as D
from trade_eda import demo, adapters


def _lin(n=50, slope=1.0, seed=1):
    rng = random.Random(seed)
    return [slope * i + rng.gauss(0, 0.01) for i in range(n)]


# --- correlation -----------------------------------------------------------

def test_pearson_perfect():
    x = _lin()
    assert C.pearson_corr(x, x) == pytest.approx(1.0)
    assert C.pearson_corr(x, [-v for v in x]) == pytest.approx(-1.0)


def test_pearson_unrelated_near_zero():
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(2000)]
    y = [rng.gauss(0, 1) for _ in range(2000)]
    assert abs(C.pearson_corr(x, y)) < 0.1


def test_pearson_constant_series_is_zero():
    assert C.pearson_corr([1.0] * 10, _lin(10)) == 0.0


def test_pearson_validates():
    with pytest.raises(ValueError):
        C.pearson_corr([1.0], [1.0])
    with pytest.raises(ValueError):
        C.pearson_corr([1.0, 2.0], [1.0])


def test_spearman_monotonic_nonlinear():
    x = [float(i) for i in range(50)]
    y = [math.exp(v / 10.0) for v in x]  # monotonic, nonlinear
    assert C.spearman_corr(x, y) == pytest.approx(1.0)
    assert C.pearson_corr(x, y) < 0.99  # pearson penalizes nonlinearity


def test_spearman_ties():
    x = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    y = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    assert C.spearman_corr(x, y) == pytest.approx(1.0)


def test_aligned_returns_truncates_and_kinds():
    closes = {"A": [100.0, 101.0, 102.0, 103.0],
              "B": [50.0, 51.0, 52.0]}  # shorter
    r = C.aligned_returns(closes, kind="simple")
    assert len(r["A"]) == len(r["B"]) == 2
    # A is truncated to the common length (oldest-first): [101, 102, 103].
    assert r["A"][0] == pytest.approx(102.0 / 101.0 - 1.0)
    rl = C.aligned_returns(closes, kind="log")
    assert rl["A"][0] == pytest.approx(math.log(102.0 / 101.0))
    with pytest.raises(ValueError):
        C.aligned_returns({"A": [1.0]}, kind="log")
    with pytest.raises(ValueError):
        C.aligned_returns({"A": [1.0, 2.0]}, kind="weird")


def test_corr_matrix_shape_symmetry():
    rets = {"A": _lin(40), "B": _lin(40, slope=-1.0), "C": _lin(40, seed=9)}
    m = C.corr_matrix(rets, method="pearson")
    assert m["symbols"] == ["A", "B", "C"]
    mat = m["matrix"]
    for i in range(3):
        assert mat[i][i] == 1.0
        for j in range(3):
            assert mat[i][j] == pytest.approx(mat[j][i])
            assert -1.0 <= mat[i][j] <= 1.0
    assert mat[0][1] < -0.9
    with pytest.raises(ValueError):
        C.corr_matrix({"A": _lin(10)})


def test_mean_and_max_pairwise():
    rets = {"A": _lin(40), "B": _lin(40, slope=1.0, seed=2),
            "C": _lin(40, seed=9)}
    m = C.corr_matrix(rets)
    mp = C.mean_pairwise_corr(m)
    assert -1.0 <= mp <= 1.0
    a, b, v = C.max_pairwise_corr(m)
    assert {a, b} == {"A", "B"} and v > 0.9


# --- covariance ------------------------------------------------------------

def _rets(n_assets=3, n=300, seed=5):
    rng = random.Random(seed)
    return {f"S{i}": [rng.gauss(0, 0.01) for _ in range(n)]
            for i in range(n_assets)}


def test_sample_cov_symmetry_and_scale():
    rets = _rets()
    cov = V.sample_cov(rets, annualize=252)
    mat = cov["matrix"]
    assert cov["symbols"] == ["S0", "S1", "S2"]
    for i in range(3):
        assert mat[i][i] == pytest.approx(0.01 ** 2 * 252, rel=0.2)
        for j in range(3):
            assert mat[i][j] == pytest.approx(mat[j][i])


def test_sample_cov_validates():
    with pytest.raises(ValueError):
        V.sample_cov({"A": [0.1]})
    with pytest.raises(ValueError):
        V.sample_cov({"A": [0.1, 0.2], "B": [0.1]})


def test_ledoit_wolf_delta_bounds():
    lw = V.ledoit_wolf(_rets(n_assets=4))
    assert 0.0 <= lw["shrinkage"] <= 1.0
    assert lw["target"] == "mu * I (scaled identity)"
    mat = lw["matrix"]
    for i in range(4):
        for j in range(4):
            assert mat[i][j] == pytest.approx(mat[j][i])


def test_ledoit_wolf_shrinks_hard_on_noise():
    # 10 assets, 12 obs: sample cov is garbage, LW should lean on target.
    lw = V.ledoit_wolf(_rets(n_assets=10, n=12, seed=11))
    assert lw["shrinkage"] > 0.5


def test_ledoit_wolf_shrinks_more_when_noisier():
    # Ordering property: wide/noisy data leans harder on the target than
    # clean diagonal data with plenty of observations.
    clean = V.ledoit_wolf(_rets(n_assets=3, n=2000, seed=13))["shrinkage"]
    noisy = V.ledoit_wolf(_rets(n_assets=10, n=12, seed=11))["shrinkage"]
    assert 0.0 <= clean <= 1.0 and 0.0 <= noisy <= 1.0
    assert noisy > clean


def test_ledoit_wolf_validates():
    with pytest.raises(ValueError):
        V.ledoit_wolf({"A": [0.1, 0.2], "B": [0.1, 0.2]})


def test_portfolio_variance():
    cov = V.sample_cov(_rets())
    w = [1 / 3, 1 / 3, 1 / 3]
    var = V.portfolio_variance(w, cov)
    assert var > 0
    assert V.portfolio_volatility(w, cov) == pytest.approx(math.sqrt(var))
    with pytest.raises(ValueError):
        V.portfolio_variance([0.5, 0.5], cov)


# --- describe --------------------------------------------------------------

def test_describe_series_known_moments():
    xs = [0.01, -0.01, 0.01, -0.01] * 25
    d = D.describe_series(xs, annualize=252)
    assert d["n"] == 100
    assert d["mean"] == pytest.approx(0.0, abs=1e-12)
    assert d["vol_annualized"] == pytest.approx(0.01 * math.sqrt(252),
                                                rel=0.01)
    assert d["skew"] == pytest.approx(0.0, abs=1e-9)
    assert d["jarque_bera"] >= 0.0
    with pytest.raises(ValueError):
        D.describe_series([0.1])


def test_describe_returns_keys():
    d = D.describe_returns(_rets(n_assets=2, n=60))
    assert set(d) == {"S0", "S1"}
    assert d["S0"]["n"] == 60


def _bar(close, volume=1000, ts="2024-01-02T00:00:00+00:00"):
    return {"symbol": "T", "timestamp": ts, "open": close, "high": close,
            "low": close, "close": close, "volume": volume}


def test_data_quality_clean():
    bars = [_bar(100 + i * 0.5) for i in range(30)]
    q = D.data_quality({"T": bars})["T"]
    assert q["clean"] and q["n_bars"] == 30


def test_data_quality_flags():
    bars = [_bar(100 + i * 0.5) for i in range(30)]
    bars[5]["volume"] = 0                       # zero-volume print
    bars[6]["close"] = bars[7]["close"] = bars[8]["close"] = 100.0  # stale run
    bars[10]["close"] = 500.0                   # MAD outlier
    bars[12]["close"] = None                    # missing
    q = D.data_quality({"T": bars})["T"]
    assert not q["clean"]
    assert q["zero_volume_bars"] == 1
    assert q["stale_close_runs"] == 1
    assert q["price_outliers_mad"] == 1
    assert q["missing_closes"] == 1


# --- demo ------------------------------------------------------------------

def test_demo_deterministic():
    a = demo.demo_universe(seed=7)
    b = demo.demo_universe(seed=7)
    assert a == b
    c = demo.demo_universe(seed=8)
    assert a["BLK1_1"] != c["BLK1_1"]


def test_demo_block_structure():
    closes = demo.demo_universe(n_days=400, seed=7)
    rets = C.aligned_returns(closes)
    m = C.corr_matrix(rets)
    idx = {s: i for i, s in enumerate(m["symbols"])}
    mat = m["matrix"]
    within = [mat[idx[f"BLK1_{i}"]][idx[f"BLK1_{j}"]]
              for i in range(1, 5) for j in range(i + 1, 5)]
    across = [mat[idx[f"BLK1_{i}"]][idx[f"BLK2_{j}"]]
              for i in range(1, 5) for j in range(1, 5)]
    assert sum(within) / len(within) > 0.4
    assert sum(across) / len(across) < 0.3


def test_demo_bars_shape():
    bars = demo.demo_bars(n_days=60, seed=7)
    assert set(bars) == {f"BLK1_{i}" for i in range(1, 5)} | \
        {f"BLK2_{i}" for i in range(1, 5)}
    b = bars["BLK1_1"][0]
    assert set(b) >= {"symbol", "timestamp", "open", "high", "low",
                      "close", "volume"}
    assert b["high"] >= max(b["open"], b["close"]) >= b["low"] >= 0


# --- analyze ---------------------------------------------------------------

def test_analyze_full_report():
    bars = demo.demo_bars(n_days=120, seed=7)
    rep = trade_eda.analyze(bars)
    assert rep["source"] == "trade-eda"
    assert len(rep["symbols"]) == 8
    assert rep["n_obs"] == 119
    assert set(rep["correlation"]["matrix"][0]) is not None
    assert rep["covariance"]["shrinkage"] == "ledoit_wolf"
    assert 0.0 <= rep["covariance"]["shrinkage_delta"] <= 1.0
    assert set(rep["describe"]) == set(rep["symbols"])
    assert set(rep["quality"]) == set(rep["symbols"])
    div = rep["diversification"]
    assert div["effective_n_equal_weight"] < 8  # correlated: fewer bets
    hi = div["max_pairwise_corr"]
    assert hi["a"].split("_")[0] == hi["b"].split("_")[0]  # within block


def test_analyze_spearman_and_sample():
    bars = demo.demo_bars(n_days=60, seed=7)
    rep = trade_eda.analyze(bars, method="spearman", shrinkage="sample",
                            kind="simple", lookback=40)
    assert rep["correlation"]["method"] == "spearman"
    assert rep["n_obs"] == 39
    assert rep["covariance"]["shrinkage_delta"] == 0.0


def test_analyze_validates():
    bars = demo.demo_bars(n_days=60, seed=7)
    with pytest.raises(ValueError):
        trade_eda.analyze({"ONLY": bars["BLK1_1"]})
    with pytest.raises(ValueError):
        trade_eda.analyze(bars, shrinkage="bogus")
    with pytest.raises(ValueError):
        trade_eda.analyze({"A": [{"close": 1.0}], "B": [{"close": 2.0}]})


# --- adapters ---------------------------------------------------------------

def test_corr_for_optimize_shape():
    out = adapters.corr_for_optimize(demo.demo_bars(n_days=60, seed=7))
    assert out["source"] == "trade-eda"
    assert len(out["symbols"]) == 8
    assert len(out["cov_annualized"]) == 8
    assert 0.0 <= out["shrinkage_delta"] <= 1.0


def test_corr_for_risk_flags():
    out = adapters.corr_for_risk(demo.demo_bars(n_days=120, seed=7))
    assert out["effective_n_equal_weight"] < 8
    assert "behave like" in out["notes"]


def test_to_agent_eda_report_shape():
    out = adapters.to_agent_eda_report(demo.demo_bars(n_days=120, seed=7))
    assert out["source"] == "trade-eda"
    assert set(out["vol_annualized"]) == set(out["symbols"])
    assert isinstance(out["nonnormal"], list)


def test_eda_forecast_inputs_shape():
    out = adapters.eda_forecast_inputs(demo.demo_bars(n_days=60, seed=7),
                                       horizon=21)
    assert out["horizon_days"] == 21
    assert out["correlation"]["method"] == "pearson"
