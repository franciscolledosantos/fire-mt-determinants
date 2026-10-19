"""Figura 5 (nova): coeficientes locais da GWNBR (k = 340) com erros-padrão locais (sanduíche GWR)
e máscara de significância ajustada para testes múltiplos (da Silva & Fotheringham 2016: alfa = 0,05 / (tr(S)/p))."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, matplotlib
from scipy import stats
from scipy.spatial import cKDTree
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import gw_avancado as G

D, W = G.D, G.W; F = os.path.join(W, "figuras")
aj = json.load(open(os.path.join(D, "gwnbr_ajuste.json"))); k, alpha, trS = aj["k_GWNBR"], aj["alpha_global"], aj["trS"]
c = pd.read_csv(os.path.join(D, "gwnbr_coef_locais.csv"))
B = c[["b_intercepto"] + [f"b_{v}" for v in G.VARS]].values
Xd, off, C = G.Xd, G.off, G.C
dd, ii = cKDTree(C).query(C, k=k)
SE = np.zeros_like(B)
for i in range(len(B)):
    idx = ii[i]; w = G.bisq(dd[i], dd[i, -1] * 1.0000001); Xi = Xd[idx]
    mu = np.exp(Xi @ B[i] + off[idx]); v = mu / (1 + alpha * mu)
    A = np.linalg.inv(Xi.T @ (Xi * (w * v)[:, None]) + 1e-6 * np.eye(Xi.shape[1]))
    M = Xi.T @ (Xi * (w ** 2 * v)[:, None])
    SE[i] = np.sqrt(np.diag(A @ M @ A))
T = B / SE
p = Xd.shape[1]; a_adj = 0.05 / (trS / p); tcrit = stats.norm.ppf(1 - a_adj / 2)
out = c[["gx", "gy", "x", "y", "bioma"]].copy()
for j, v in enumerate(["intercepto"] + G.VARS):
    out[f"b_{v}"] = B[:, j]; out[f"se_{v}"] = SE[:, j]; out[f"t_{v}"] = T[:, j]
out.to_csv(os.path.join(D, "gwnbr_coef_se_locais.csv"), index=False)
res = []
for v in G.VARS:
    for bname, g in out.groupby("bioma"):
        res.append({"variavel": v, "bioma": bname, "mediana": g[f"b_{v}"].median(), "q10": g[f"b_{v}"].quantile(.1), "q90": g[f"b_{v}"].quantile(.9),
                    "frac_sig_pos": (g[f"t_{v}"] > tcrit).mean(), "frac_sig_neg": (g[f"t_{v}"] < -tcrit).mean()})
R = pd.DataFrame(res); R.round(3).to_csv(os.path.join(D, "gwnbr_resumo_bioma.csv"), index=False)
print(f"t crítico ajustado = {tcrit:.2f} (alfa {a_adj:.4f})"); print(R.round(2).to_string())

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; BASE = "#c3c2b7"; GRID = "#e1e0d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 300})
div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#f6b8a8", "#e34948", "#a32524"])
itens = [("conv", "a  Recent native-vegetation conversion"), ("floresta", "b  Forest fraction"),
         ("p_chuvosa_pre", "c  Wet-season rainfall (climatology)"), ("cdd", "d  Consecutive dry days (climatology)")]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6))
for ax, (v, t) in zip(axs.ravel(), itens):
    b = out[f"b_{v}"].values; sig = np.abs(out[f"t_{v}"].values) > tcrit; lim = np.quantile(np.abs(b), 0.98)
    ax.scatter(out.x[~sig], out.y[~sig], c=GRID, s=9, marker="s", linewidths=0)
    sc = ax.scatter(out.x[sig], out.y[sig], c=b[sig], cmap=div, norm=TwoSlopeNorm(0, -lim, lim), s=9, marker="s", linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("local coefficient (per SD)", color=INK2, fontsize=7)
    ax.set_title(t, loc="left", color=INK, fontsize=9); ax.set_aspect("equal"); ax.tick_params(labelsize=6, colors=MUTED, length=2); ax.set_xlabel("Longitude (°)", fontsize=7); ax.set_ylabel("Latitude (°)", fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(F, "fig5_gwnbr_coeficientes.png"), facecolor="white", bbox_inches="tight", pad_inches=0.05)
