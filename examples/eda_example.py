"""End-to-end trade-eda walkthrough: demo universe -> report -> adapters."""

from trade_eda import analyze, demo_bars
from trade_eda import adapters

bars = demo_bars(n_days=250, seed=7)
rep = analyze(bars)

print("symbols:", rep["symbols"])
print("observations:", rep["n_obs"])
print()
print("correlation (first 3x3, pearson):")
for row in rep["correlation"]["matrix"][:3]:
    print("  ", [round(v, 2) for v in row[:3]])
print()
print("diversification:", rep["diversification"])
print()
print("shrinkage delta:", rep["covariance"]["shrinkage_delta"])
print()
print("annualized vol:")
for s, d in rep["describe"].items():
    print(f"  {s}: {d['vol_annualized']:.1%}")
print()
print("quality flags:",
      {s: q for s, q in rep["quality"].items() if not q["clean"]}
      or "all clean")
print()
print("risk adapter:", adapters.corr_for_risk(bars)["notes"])
print()
print("agent brief keys:", sorted(adapters.to_agent_eda_report(bars)))
