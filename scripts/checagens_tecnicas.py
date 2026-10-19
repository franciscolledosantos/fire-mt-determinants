"""Checagens técnicas pendentes: VIF dos fatores, I de Moran dos resíduos (NB-GLM e LightGBM fora do fold)
por célula, superdispersão e zeros observados × previstos pela NB."""
import os, sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import sparse
from glum import GeneralizedLinearRegressor
from cv_modelos import preparar, D, LU, CLIM, STAT
from cv_lib import folds_spatial

pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet"))
pan, Xg, Xb = preparar(pan); y = pan.focos.values.astype(float)
out = {}
# 1) VIF (fatores principais, sem interações)
Z = pan[LU + CLIM + STAT + ["conv"]].copy()
Z = (Z - Z.mean()) / Z.std()
C = np.corrcoef(Z.values, rowvar=False)
vif = np.diag(np.linalg.inv(C))
out["vif"] = {c: round(float(v), 2) for c, v in sorted(zip(Z.columns, vif), key=lambda t: -t[1])}
print("VIF", out["vif"], flush=True)
# 2) NB-GLM em todos os dados + LightGBM fora do fold (blocos de 1°)
glm = GeneralizedLinearRegressor(family="negative.binomial", alpha=1e-4, l1_ratio=0).fit(Xg.values, y)
mu_glm = glm.predict(Xg.values)
mu_gbm = np.full(len(y), np.nan)
p = dict(num_leaves=63, min_child_samples=50, learning_rate=0.05, n_estimators=400, colsample_bytree=0.8, subsample=0.8,
         subsample_freq=1, reg_lambda=0, objective="poisson", verbose=-1, n_jobs=4)
for tr, te in folds_spatial(pan.x.values, pan.y.values, 1.0, 5):
    mu_gbm[te] = lgb.LGBMRegressor(**p).fit(Xb.iloc[tr], y[tr]).predict(Xb.iloc[te])
# 3) superdispersão e zeros
alpha = 1.0  # NB do glum usa theta=1 por padrão: estimar alpha por momentos
r = (y - mu_glm)
alpha_m = float(np.mean((r ** 2 - mu_glm) / mu_glm ** 2))
th = 1 / max(alpha_m, 1e-6)
p0_nb = (th / (th + mu_glm)) ** th
out["dispersao"] = {"media": float(y.mean()), "variancia": float(y.var()), "alpha_momentos_glm": alpha_m,
                    "zeros_obs": float((y == 0).mean()), "zeros_prev_poisson_glm": float(np.mean(np.exp(-mu_glm))),
                    "zeros_prev_nb_glm": float(np.mean(p0_nb))}
print(out["dispersao"], flush=True)
# 4) I de Moran dos resíduos de Pearson médios por célula (vizinhança de torre na grade de 0,05°)
def moran(v, cy, cx):
    idx = {(a, b): i for i, (a, b) in enumerate(zip(cy, cx))}
    rows, cols = [], []
    for i, (a, b) in enumerate(zip(cy, cx)):
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = idx.get((a + da, b + db))
            if j is not None:
                rows.append(i); cols.append(j)
    Wm = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(v), len(v)))
    z = v - v.mean(); S0 = Wm.sum()
    I = len(v) / S0 * (z @ (Wm @ z)) / (z @ z)
    rng = np.random.default_rng(0)
    perm = [ (lambda zz: len(v) / S0 * (zz @ (Wm @ zz)) / (zz @ zz))(rng.permutation(z)) for _ in range(199)]
    return float(I), float((np.sum(np.array(perm) >= I) + 1) / 200)
cel = pan.assign(rg=(y - mu_glm) / np.sqrt(mu_glm + alpha_m * mu_glm ** 2),
                 rb=(y - mu_gbm) / np.sqrt(mu_gbm + alpha_m * mu_gbm ** 2),
                 ry=y).groupby(["cy", "cx"])[["rg", "rb", "ry"]].mean().reset_index()
out["moran"] = {}
for nome, col in (("focos médios (sem modelo)", "ry"), ("resíduo NB-GLM", "rg"), ("resíduo LightGBM (fora do fold)", "rb")):
    I, pv = moran(cel[col].values, cel.cy.values, cel.cx.values)
    out["moran"][nome] = {"I": round(I, 3), "p_perm": pv}
    print(nome, I, pv, flush=True)
json.dump(out, open(os.path.join(D, "checagens_tecnicas.json"), "w"), indent=1, ensure_ascii=False)
