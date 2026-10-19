"""Figura 7: validação espacial dos métodos avançados.
a) painel 0,05° × ano, blocos de 1° (5 folds): NB-GLM, ZINB espacial BYM, LightGBM
b) células de 0,25°, blocos de 2° (3 × 5 folds): NB global, GWPR, GWNBR, RF global, GWRF, LightGBM"""
import os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados"); F = os.path.join(W, "figuras")
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"; ACC = "#2a78d6"; GW = "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
r = pd.read_csv(os.path.join(D, "cv_resultados_folds_final.csv")); r = r[r.esquema.str.contains("1°")]
b = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, "bym_cv_fold*.csv")))])
A = pd.concat([r[r.modelo.isin(["NB-GLM", "LightGBM (padrão)"])][["modelo", "D2"]], b[["modelo", "D2"]]])
A["modelo"] = A.modelo.replace({"NB-GLM": "NB-GLM", "LightGBM (padrão)": "LightGBM", "ZINB espacial BYM": "ZINB + BYM spatial effect"})
ordA = ["NB-GLM", "ZINB + BYM spatial effect", "LightGBM"]
g = pd.read_csv(os.path.join(D, "gw_avancado_cv_folds.csv")); g = g[g.modelo.isin(["NB global", "GWPR", "GWNBR", "RF global", "GWRF (a=0.5)", "LightGBM"])]
g["modelo"] = g.modelo.replace({"GWRF (a=0.5)": "GWRF (a = 0.5)"})
ordB = ["NB global", "GWNBR", "GWPR", "RF global", "LightGBM", "GWRF (a = 0.5)"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.4]})
for ax, df, ordm, title, gwset in [(axs[0], A, ordA, "a  0.05° cell-years, 1° spatial blocks", set()),
                                     (axs[1], g, ordB, "b  0.25° cells, 2° spatial blocks", {"GWNBR", "GWPR", "GWRF (a = 0.5)"})]:
    for i, m in enumerate(ordm):
        v = df[df.modelo == m].D2.values; c = GW if m in gwset else (ACC if "BYM" in m else MUTED)
        ax.scatter(np.clip(v, -0.2, None), np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), s=10, color=c, alpha=0.5, linewidths=0)
        ax.plot([np.median(v)] * 2, [i - 0.28, i + 0.28], color=INK if c == MUTED else c, lw=2.2)
        ax.text(1.0, i, f"{np.median(v):.2f}", transform=ax.get_yaxis_transform(), va="center", ha="left", color=INK2, fontsize=7)
    ax.set_yticks(range(len(ordm))); ax.set_yticklabels(ordm); ax.set_xlabel("Deviance explained in held-out blocks (D²)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_title(title, loc="left", color=INK, fontsize=9); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.2 if ax is axs[1] else 0.05, 0.85 if ax is axs[1] else 0.42)
fig.text(0.5, -0.02, "Bars: median across folds; points: individual folds (values below −0.2 clipped). Orange: geographically weighted models.", ha="center", color=MUTED, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig7_validacao_avancados.png"), facecolor="white", bbox_inches="tight", pad_inches=0.05)
