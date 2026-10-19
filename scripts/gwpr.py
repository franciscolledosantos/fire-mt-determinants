"""Regressão de Poisson geograficamente ponderada (GWPR) para focos em MT.

Agrega o painel 0,05° × ano para células de 0,25° (5×5) e soma 2010–2024.
Resposta: focos totais; offset: log(área km² × anos). Covariáveis: frações de uso do solo médias (t-1),
conversão média, CDD médio, chuva da estação chuvosa média, rodovias, população, relevo.

1) Poisson global × GWPR (mgwr; banda adaptativa bisquare selecionada por AICc).
2) Coeficientes locais e teste de não estacionariedade (Monte Carlo do mgwr não é rodado — custo; usamos
   a dispersão interquartil dos coeficientes locais versus o erro padrão global como diagnóstico).
3) Validação por blocos espaciais de 2°: GWPR (ajuste local fora do bloco) × Poisson global × LightGBM.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from spglm.family import Poisson
from scipy.spatial import cKDTree
from cv_lib import metrics, folds_spatial
import lightgbm as lgb

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados")
VARS = ["floresta", "savana", "campestre", "alagado", "soja", "outras_temp", "conv", "cdd", "p_chuvosa_pre", "lroad", "lpop", "slope"]

pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet"))
pan["area"] = (0.05 * 111.32) ** 2 * np.cos(np.deg2rad(pan.y))
pan["conv"] = pan.conversao.fillna(0)
pan["gx"] = np.floor(pan.x / 0.25).astype(int); pan["gy"] = np.floor(pan.y / 0.25).astype(int)
ag = pan.groupby(["gx", "gy"]).agg(focos=("focos", "sum"), area=("area", "sum"), n=("focos", "size"),
                                    x=("x", "mean"), y=("y", "mean"),
                                    bioma=("bioma", lambda s: s.mode().iloc[0]),
                                    **{v: (v, "mean") for v in ["floresta", "savana", "campestre", "alagado", "soja", "outras_temp", "conv", "cdd", "p_chuvosa_pre", "roaddist", "denspop", "slope"]})
ag = ag[ag.n >= 15 * 10].reset_index()  # células de 0,25° com pelo menos 10 células finas
ag["anos"] = 15; ag["area"] = ag.area / 15
ag["lroad"] = np.log1p(ag.roaddist); ag["lpop"] = np.log1p(ag.denspop * 1000)
ag["conv"] = ag.conv * 100  # p.p./ano
Xr = ag[VARS].values
mu_, sd_ = Xr.mean(0), Xr.std(0); X = (Xr - mu_) / sd_
y = ag.focos.values.reshape(-1, 1).astype(float)
off = np.log(ag.area.values * ag.anos.values).reshape(-1, 1)
# coordenadas projetadas aproximadas (km)
coords = np.c_[ag.x.values * 111.32 * np.cos(np.deg2rad(-12.5)), ag.y.values * 111.32]
print("células 0,25°:", len(ag), "focos:", int(y.sum()), flush=True)

# --- Poisson global
glob_m = GWR(coords, y, X, bw=len(y), family=Poisson(), offset=off, kernel="bisquare", fixed=False)
# bw = n com bisquare ainda pondera; para o global puro uso statsmodels-like IRLS do spglm
from spglm.glm import GLM
g = GLM(y, X, family=Poisson(), offset=off).fit()
print("Poisson global AIC:", round(g.aic, 1), flush=True)

# --- GWPR
t0 = time.time()
sel = Sel_BW(coords, y, X, family=Poisson(), offset=off, kernel="bisquare", fixed=False)
bw = sel.search(criterion="AICc", bw_min=150, bw_max=len(y) - 1)
res = GWR(coords, y, X, bw=bw, family=Poisson(), offset=off, kernel="bisquare", fixed=False).fit()
print(f"GWPR banda adaptativa = {bw} vizinhos ({time.time()-t0:.0f}s); AICc GWPR {res.aicc:.1f}; AIC global {g.aic:.1f}", flush=True)
loc = pd.DataFrame(res.params, columns=["intercepto"] + VARS)
tv = pd.DataFrame(res.filter_tvals(), columns=["intercepto"] + VARS)
out = pd.concat([ag[["gx", "gy", "x", "y", "bioma", "focos"]], loc.add_prefix("b_"), tv.add_prefix("t_")], axis=1)
out.to_csv(os.path.join(D, "gwpr_coef_locais.csv"), index=False)
resumo = pd.DataFrame({"global": np.r_[g.params], "global_ep": np.r_[g.bse],
                       "local_q25": loc.quantile(.25).values, "local_mediana": loc.median().values, "local_q75": loc.quantile(.75).values,
                       "frac_sig_pos": ((tv > 0) & (tv != 0)).mean().values, "frac_sig_neg": ((tv < 0) & (tv != 0)).mean().values},
                      index=["intercepto"] + VARS)
resumo["IQR/(2·EP global)"] = (resumo.local_q75 - resumo.local_q25) / (2 * resumo.global_ep)
resumo.round(3).to_csv(os.path.join(D, "gwpr_resumo.csv")); print(resumo.round(3).to_string(), flush=True)
porbioma = loc.groupby(ag.bioma).median().T; porbioma.round(3).to_csv(os.path.join(D, "gwpr_mediana_bioma.csv")); print(porbioma.round(3).to_string(), flush=True)
json.dump({"bw": int(bw), "aicc_gwpr": float(res.aicc), "aic_global": float(g.aic), "n": int(len(y)),
           "pD2_global": float(np.ravel(1 - g.deviance / g.null_deviance)[0]), "pD2_gwpr": float(np.ravel(res.pDev)[0])}, open(os.path.join(D, "gwpr_ajuste.json"), "w"), indent=1)


# --- validação por blocos espaciais de 2°
def poisson_irls(Xd, yv, offv, w=None, it=50):
    Xd = np.c_[np.ones(len(Xd)), Xd]; b = np.zeros(Xd.shape[1]); b[0] = np.log(yv.sum() / np.exp(offv).sum())
    w = np.ones(len(yv)) if w is None else w
    for _ in range(it):
        eta = Xd @ b + offv; mu = np.exp(np.clip(eta, -30, 30)); z = eta - offv + (yv - mu) / mu
        WW = w * mu; A = Xd.T @ (Xd * WW[:, None]) + 1e-6 * np.eye(Xd.shape[1]); bn = np.linalg.solve(A, Xd.T @ (WW * z))
        if np.max(np.abs(bn - b)) < 1e-8: b = bn; break
        b = bn
    return b


def gwpr_predict(ctr, Xtr, ytr, otr, cte, Xte, ote, k):
    tree = cKDTree(ctr); pred = np.empty(len(cte))
    d, idx = tree.query(cte, k=min(k, len(ctr)))
    for i in range(len(cte)):
        h = d[i, -1] * 1.0000001; w = (1 - (d[i] / h) ** 2) ** 2
        b = poisson_irls(Xtr[idx[i]], ytr[idx[i]], otr[idx[i]], w)
        pred[i] = np.exp(b[0] + Xte[i] @ b[1:] + ote[i])
    return pred


yv, ov = y.ravel(), off.ravel()
rows = []
for rep in range(3):
    for fi, (tr, te) in enumerate(folds_spatial(ag.x.values, ag.y.values, 2.0, 5, seed=rep)):
        bg = poisson_irls(X[tr], yv[tr], ov[tr]); pg = np.exp(np.c_[np.ones(len(te)), X[te]] @ bg + ov[te])
        pw = gwpr_predict(coords[tr], X[tr], yv[tr], ov[tr], coords[te], X[te], ov[te], int(bw))
        gb = lgb.LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1, n_jobs=2)
        gb.fit(np.c_[X[tr]], yv[tr], init_score=ov[tr]); pb = np.exp(gb.predict(X[te], raw_score=True) + ov[te])
        for nm, p in [("Poisson global", pg), ("GWPR", pw), ("LightGBM", pb)]:
            m = metrics(yv[te], p); m.update(modelo=nm, rep=rep, fold=fi); rows.append(m)
    print("rep", rep, "ok", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "gwpr_cv_folds.csv"), index=False)
S = R.groupby("modelo")[["D2", "Spearman", "MAE"]].agg(["mean", "std"]).round(3); S.to_csv(os.path.join(D, "gwpr_cv_resumo.csv")); print(S.to_string())
