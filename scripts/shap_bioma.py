"""SHAP do LightGBM (hiperparâmetros escolhidos com CV interna espacial) por bioma.

Saídas: importância média |SHAP| por bioma, curvas de dependência das variáveis-chave por bioma (figura 6)
e a interação chuva jun–set × fração de floresta (valores de interação SHAP numa subamostra).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, lightgbm as lgb, shap, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cv_modelos import preparar, D, W

F = os.path.join(W, "figuras")
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet")); pan, Xg, Xb = preparar(pan); y = pan.focos.values
p = json.load(open(os.path.join(D, "hiperparametros_escolhidos.json")))["interno_espacial"]
MF = os.path.join(D, "lgbm_shap_modelo.txt")
if os.path.exists(MF):
    bst = lgb.Booster(model_file=MF)
else:
    bst = lgb.LGBMRegressor(objective="poisson", verbose=-1, n_jobs=2, subsample_freq=1, **p).fit(Xb, y).booster_; bst.save_model(MF)
print("modelo pronto", flush=True)
rng = np.random.default_rng(0)
idx = np.concatenate([rng.choice(np.where(pan.bioma == b)[0], 8000, replace=False) for b in ["Amazonia", "Cerrado", "Pantanal"]])
Xs = Xb.iloc[idx].reset_index(drop=True); bs = pan.bioma.values[idx]
CF = os.path.join(D, "shap_valores_amostra.npy")
if os.path.exists(CF): sv = np.load(CF)
else: sv = bst.predict(Xs, pred_contrib=True); np.save(CF, sv)   # TreeSHAP nativo do LightGBM (última coluna = valor base)
S = pd.DataFrame(sv[:, :-1], columns=Xs.columns)
imp = S.abs().groupby(bs).mean().T; imp["geral"] = S.abs().mean(); imp = imp.sort_values("geral", ascending=False)
imp.round(4).to_csv(os.path.join(D, "shap_importancia_bioma.csv")); print(imp.round(3).to_string(), flush=True)

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRID = "#e1e0d9"; BASE = "#c3c2b7"
BIO = {"Amazonia": "#2a78d6", "Cerrado": "#eb6834", "Pantanal": "#1baf7a"}; NOME = {"Amazonia": "Amazon", "Cerrado": "Cerrado", "Pantanal": "Pantanal"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "savefig.dpi": 300})
VARS = [("p_jjas_anom", "Dry-season rain anomaly (SD)"), ("p_chuvosa_pre_anom", "Preceding wet-season rain anomaly (SD)"), ("cdd_anom", "CDD anomaly (SD)"),
        ("conv", "log(1 + p.p. converted)"), ("floresta", "Forest fraction"), ("soja", "Soybean fraction")]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey=False)
for ax, (v, lab) in zip(axs.ravel(), VARS):
    for b in BIO:
        mk = bs == b; xv = Xs[v].values[mk]; sv_ = S[v].values[mk]
        q = np.quantile(xv, np.linspace(0.02, 0.98, 25)); qs = np.unique(q)
        if len(qs) < 4:
            ax.scatter(xv, sv_, s=1, color=BIO[b], alpha=0.15, rasterized=True); continue
        cut = np.digitize(xv, qs); mx = [xv[cut == k].mean() for k in range(1, len(qs)) if (cut == k).sum() > 20]
        my = [sv_[cut == k].mean() for k in range(1, len(qs)) if (cut == k).sum() > 20]
        ax.scatter(xv, sv_, s=1, color=BIO[b], alpha=0.06, rasterized=True, linewidths=0)
        ax.plot(mx, my, color=BIO[b], lw=2, label=NOME[b])
    ax.axhline(0, color=BASE, lw=1); ax.set_xlabel(lab); ax.set_ylabel("SHAP (log rate)")
axs[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK2)
for i, a in enumerate(axs.ravel()): a.set_title("abcdef"[i], loc="left", color=INK, fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(F, "fig6_shap_bioma.png"), facecolor="white", bbox_inches="tight", pad_inches=0.05)

# interação jun–set × floresta: SHAP da chuva jun–set por classe de floresta, em anos secos (anomalia < −0,5 DP) e normais
inter = pd.DataFrame({"bioma": bs, "jjas": Xs.p_jjas_anom.values, "floresta": Xs.floresta.values, "shap_jjas": S.p_jjas_anom.values})
tab = inter.assign(seco=np.where(inter.jjas < -0.5, "seco", "normal/úmido"), flor=pd.cut(inter.floresta, [-0.01, 0.2, 0.6, 1.0], labels=["<20%", "20–60%", ">60%"])) \
    .groupby(["bioma", "seco", "flor"]).shap_jjas.mean().unstack().round(3)
tab.to_csv(os.path.join(D, "shap_interacao_jjas_floresta.csv")); print(tab.to_string())
