"""Revisão 17/10/2026 — figuras na numeração da JEM.
Fig. 2 nova: focos e área queimada por bioma (2010–2025, 2025 fora do ajuste) e as três anomalias
climáticas (estação seca jun–set, estação chuvosa precedente out–abr, CDD) — funde a Fig. 2 e a Fig. 6
da versão anterior. As demais figuras são copiadas de fogo_final/figures (numeração Fire Ecology) para
fogo_final/jem/figures com a numeração da JEM: Fig1 = Fig1, Fig2 = nova, Fig3 = Fig3, Fig4 = Fig4,
Fig5 = Fig6 (validação), S1 = Fig5 (SHAP), S2 = Fig7 (validação avançada)."""
import os, shutil, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados")
SRC = os.path.join(W, "figuras", "fire_ecology_numbering"); OUT = os.path.join(W, "figuras")
os.makedirs(OUT, exist_ok=True)
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"; SURF = "#ffffff"
BIO = {"Amazonia": "#2a78d6", "Cerrado": "#eb6834", "Pantanal": "#1baf7a"}; NOME = {"Amazonia": "Amazon", "Cerrado": "Cerrado", "Pantanal": "Pantanal"}
MK = {"Amazonia": "o", "Cerrado": "s", "Pantanal": "^"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "figure.dpi": 300, "savefig.dpi": 300, "axes.formatter.use_locale": False})
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025_fogo.parquet"))
T = pan.groupby(["year", "bioma"]).agg(focos=("focos", "sum"), queim=("queim_ha", "sum")).reset_index()
A = pan.groupby(["year", "bioma"])[["p_jjas_anom", "p_chuvosa_pre_anom", "cdd_anom"]].mean().reset_index()
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.28)
axA = fig.add_subplot(gs[0:3, 0]); axB = fig.add_subplot(gs[3:6, 0], sharex=axA)
axC = fig.add_subplot(gs[0:2, 1], sharex=axA); axD = fig.add_subplot(gs[2:4, 1], sharex=axA); axE = fig.add_subplot(gs[4:6, 1], sharex=axA)
panels = [(axA, "a", "Active fires (thousand)"), (axB, "b", "Burned area (thousand ha)"), (axC, "c", "Dry-season rainfall\n(Jun–Sep) anomaly (SD)"),
          (axD, "d", "Preceding wet-season\nrainfall anomaly (SD)"), (axE, "e", "Consecutive dry days\nanomaly (SD)")]
for b in BIO:
    t = T[T.bioma == b].sort_values("year"); a = A[A.bioma == b].sort_values("year")
    kw = dict(color=BIO[b], lw=1.6, marker=MK[b], ms=3.6, mec=SURF, mew=0.5)
    for ax, s in [(axA, t.focos / 1e3), (axB, t.queim / 1e3), (axC, a.p_jjas_anom), (axD, a.p_chuvosa_pre_anom), (axE, a.cdd_anom)]:
        yr = t.year.values if ax in (axA, axB) else a.year.values
        ax.plot(yr[yr <= 2024], s.values[yr <= 2024], **kw)
        ax.plot(yr[yr >= 2024], s.values[yr >= 2024], ls=(0, (2, 1.5)), **{k: v for k, v in kw.items() if k != "marker"})
        ax.plot([2025], [s.values[yr == 2025][0]], marker=MK[b], ms=3.8, color=SURF, mec=BIO[b], mew=1.0, ls="none")
for ax, lab, yl in panels:
    ax.set_title(lab, loc="left", color=INK, fontsize=9, fontweight="bold"); ax.set_ylabel(yl, fontsize=7.5)
    ax.axvspan(2024.5, 2025.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xticks(range(2010, 2026, 3)); ax.tick_params(labelsize=7)
    ax.grid(axis="x", visible=False)
for ax in (axC, axD, axE): ax.axhline(0, color=BASE, lw=0.8, zorder=1)
for ax in (axA, axC, axD): plt.setp(ax.get_xticklabels(), visible=False)
axB.set_xlabel("Year"); axE.set_xlabel("Year")
axA.annotate("Pantanal 2020", xy=(2020, T[(T.bioma == "Pantanal") & (T.year == 2020)].focos.values[0] / 1e3), xytext=(2014.6, 10.4), color=INK2, fontsize=7,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
axA.set_ylim(top=axA.get_ylim()[1] * 1.22)
axB.text(2025, axB.get_ylim()[1] * 0.98, "2025\nheld\nout", ha="center", va="top", fontsize=6.3, color=INK2)
h = [Line2D([], [], color=BIO[b], marker=MK[b], ms=4, mec=SURF, lw=1.6, label=NOME[b]) for b in BIO]
axA.legend(handles=h, frameon=False, ncol=3, loc="upper center", fontsize=7, labelcolor=INK2, handlelength=1.8, columnspacing=1.0)
fig.savefig(os.path.join(OUT, "Fig2.png"), facecolor=SURF, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
# valores da figura, para conferência
T.pivot(index="year", columns="bioma", values=["focos", "queim"]).round(0).to_csv(os.path.join(D, "fig2_focos_area_bioma_ano.csv"))
A.round(2).to_csv(os.path.join(D, "fig2_anomalias_bioma_ano.csv"), index=False)
# demais figuras: renumeração JEM
for src, dst in [("Fig1.png", "Fig1.png"), ("Fig3.png", "Fig3.png"), ("Fig4.png", "Fig4.png"), ("Fig6.png", "Fig5.png"), ("Fig5.png", "FigS1.png"), ("Fig7.png", "FigS2.png")]:
    shutil.copyfile(os.path.join(SRC, src), os.path.join(OUT, dst))
print("ok", sorted(os.listdir(OUT)))
