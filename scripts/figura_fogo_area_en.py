"""Figure 8 (English): focos de calor (INPE) e área queimada (MapBiomas Fogo C5) por bioma, 2010–2025."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados"); F = os.path.join(W, "figuras")
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
BIO = {"Amazonia": "#2a78d6", "Cerrado": "#eb6834", "Pantanal": "#1baf7a"}; NOME = {"Amazonia": "Amazon", "Cerrado": "Cerrado", "Pantanal": "Pantanal"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "savefig.dpi": 300,
                     "axes.formatter.use_locale": False})
t = pd.read_csv(os.path.join(D, "fogo_totais_bioma_ano.csv"))
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
fmt = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", " "))
for k, b in enumerate(BIO):
    s = t[t.bioma == b].sort_values("year")
    for r, (col, lab) in enumerate([("focos", "Active fires (thousand)"), ("queim_mil_ha", "Burned area (thousand ha)")]):
        ax = axs[r, k]; v = s[col] / (1e3 if col == "focos" else 1)
        ax.bar(s.year[s.year < 2025], v[s.year < 2025], color=BIO[b], width=0.75)
        ax.bar(s.year[s.year == 2025], v[s.year == 2025], color="white", edgecolor=BIO[b], hatch="////", linewidth=1, width=0.75)
        ax.yaxis.set_major_formatter(fmt); ax.grid(axis="x", visible=False)
        if k == 0: ax.set_ylabel(lab)
        if r == 0: ax.set_title(NOME[b], loc="left", color=INK, fontsize=9)
        ax.set_xticks([2010, 2015, 2020, 2025])
fig.tight_layout(); fig.savefig(os.path.join(F, "fig8_active_fires_burned_area.png"), facecolor="white", bbox_inches="tight", pad_inches=0.05)
