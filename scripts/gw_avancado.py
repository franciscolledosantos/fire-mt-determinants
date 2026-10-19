"""Modelos geograficamente ponderados avançados para focos em MT (células de 0,25°, 2010–2024).

- NB global e GWNBR (binomial negativa geograficamente ponderada; da Silva & Rodrigues 2014, versão com
  dispersão global): betas locais por IRLS ponderado com núcleo bisquare adaptativo; alpha estimado no
  modelo global. Banda escolhida por AICc (traço da matriz chapéu) numa grade.
- RF global e GWRF (random forest geograficamente ponderada; Georganos et al. 2021): floresta local com os k
  vizinhos de treino de cada ponto + mistura com a floresta global (peso local a escolhido na validação).
- Comparação com GWPR e LightGBM na mesma validação por blocos espaciais de 2° (3 repetições × 5 folds).
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.special import gammaln
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from cv_lib import metrics, folds_spatial

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados")
VARS = ["floresta", "savana", "campestre", "alagado", "soja", "outras_temp", "conv", "cdd", "p_chuvosa_pre", "lroad", "lpop", "slope"]

# ---------------------------------------------------------------- dados agregados (igual a gwpr.py)
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet"))
pan["area"] = (0.05 * 111.32) ** 2 * np.cos(np.deg2rad(pan.y)); pan["conv"] = pan.conversao.fillna(0)
pan["gx"] = np.floor(pan.x / 0.25).astype(int); pan["gy"] = np.floor(pan.y / 0.25).astype(int)
ag = pan.groupby(["gx", "gy"]).agg(focos=("focos", "sum"), area=("area", "sum"), n=("focos", "size"), x=("x", "mean"), y=("y", "mean"),
                                    bioma=("bioma", lambda s: s.mode().iloc[0]),
                                    **{v: (v, "mean") for v in ["floresta", "savana", "campestre", "alagado", "soja", "outras_temp", "conv", "cdd", "p_chuvosa_pre", "roaddist", "denspop", "slope"]})
ag = ag[ag.n >= 150].reset_index(); ag["area"] = ag.area / 15
ag["lroad"] = np.log1p(ag.roaddist); ag["lpop"] = np.log1p(ag.denspop * 1000); ag["conv"] = np.log1p(ag.conv * 100)
Xr = ag[VARS].values; X = (Xr - Xr.mean(0)) / Xr.std(0)
y = ag.focos.values.astype(float); off = np.log(ag.area.values * 15)
C = np.c_[ag.x.values * 111.32 * np.cos(np.deg2rad(-12.5)), ag.y.values * 111.32]
n, p = X.shape; Xd = np.c_[np.ones(n), X]
print("células:", n, flush=True)


# ---------------------------------------------------------------- NB2 por IRLS (alpha fixo) e alpha global por perfil
def nb_irls(Xd, y, off, alpha, w=None, it=60, b0=None):
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(Xd.shape[1]) if b0 is None else b0.copy()
    if b0 is None: b[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(it):
        eta = Xd @ b + off; mu = np.exp(np.clip(eta, -30, 30))
        v = mu / (1 + alpha * mu)                      # peso IRLS da NB2 com ligação log
        z = eta - off + (y - mu) / mu
        A = Xd.T @ (Xd * (w * v)[:, None]) + 1e-6 * np.eye(Xd.shape[1])
        bn = np.linalg.solve(A, Xd.T @ (w * v * z))
        if np.max(np.abs(bn - b)) < 1e-8: b = bn; break
        b = bn
    return b


def nb_ll(y, mu, alpha):
    r = 1 / alpha
    return np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))


def nb_global(Xd, y, off):
    best = None
    for a in np.exp(np.linspace(np.log(0.01), np.log(5), 40)):
        b = nb_irls(Xd, y, off, a); mu = np.exp(Xd @ b + off); ll = nb_ll(y, mu, a)
        if best is None or ll > best[2]: best = (b, a, ll)
    return best


def bisq(d, h):
    return np.where(d < h, (1 - (d / h) ** 2) ** 2, 0.0)


def gwnbr_fit(C, Xd, y, off, alpha, k):
    tree = cKDTree(C); dd, ii = tree.query(C, k=k)
    B = np.zeros((len(y), Xd.shape[1])); hat = np.zeros(len(y)); bg = nb_irls(Xd, y, off, alpha)
    for i in range(len(y)):
        h = dd[i, -1] * 1.0000001; w = bisq(dd[i], h); idx = ii[i]
        b = nb_irls(Xd[idx], y[idx], off[idx], alpha, w, b0=bg); B[i] = b
        mu = np.exp(Xd[idx] @ b + off[idx]); v = mu / (1 + alpha * mu)
        A = Xd[idx].T @ (Xd[idx] * (w * v)[:, None]) + 1e-6 * np.eye(Xd.shape[1])
        xi = Xd[i]; mui = np.exp(xi @ b + off[i]); vi = mui / (1 + alpha * mui)
        hat[i] = xi @ np.linalg.solve(A, xi) * vi            # w_ii = 1 no próprio ponto
    mu = np.exp(np.sum(Xd * B, 1) + off); ll = nb_ll(y, mu, alpha); trS = hat.sum()
    aicc = -2 * ll + 2 * trS + 2 * trS * (trS + 1) / max(len(y) - trS - 1, 1)
    return B, mu, ll, trS, aicc


def gwnbr_predict(Ctr, Xtr, ytr, otr, Cte, Xte, ote, alpha, k):
    tree = cKDTree(Ctr); dd, ii = tree.query(Cte, k=min(k, len(Ctr))); bg = nb_irls(Xtr, ytr, otr, alpha); pr = np.empty(len(Cte))
    for i in range(len(Cte)):
        w = bisq(dd[i], dd[i, -1] * 1.0000001); idx = ii[i]
        b = nb_irls(Xtr[idx], ytr[idx], otr[idx], alpha, w, b0=bg); pr[i] = np.exp(Xte[i] @ b + ote[i])
    return pr


def poisson_like_predict(Ctr, Xtr, ytr, otr, Cte, Xte, ote, k):  # GWPR = GWNBR com alpha ~ 0
    return gwnbr_predict(Ctr, Xtr, ytr, otr, Cte, Xte, ote, 1e-8, k)


# ---------------------------------------------------------------- RF / GWRF (alvo: taxa por km²·ano, peso = exposição)
def rf_fit(Xtr, ytr, otr, seed=0, n_est=300, msl=5, mf=0.5):
    rate = ytr / np.exp(otr)
    m = RandomForestRegressor(n_estimators=n_est, min_samples_leaf=msl, max_features=mf, n_jobs=2, random_state=seed)
    m.fit(Xtr, rate, sample_weight=np.exp(otr) / np.exp(otr).mean()); return m


def gwrf_predict(Ctr, Xtr, ytr, otr, Cte, Xte, ote, k, a_local, glob, seed=0):
    tree = cKDTree(Ctr); dd, ii = tree.query(Cte, k=min(k, len(Ctr)))
    pg = glob.predict(Xte); pl = np.empty(len(Cte))
    # agrupa pontos de teste com o mesmo conjunto de vizinhos aproximado: floresta local por ponto (custo controlado com 100 árvores)
    for i in range(len(Cte)):
        idx = ii[i]
        m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=2, random_state=seed)
        m.fit(Xtr[idx], ytr[idx] / np.exp(otr[idx]), sample_weight=bisq(dd[i], dd[i, -1] * 1.0000001) + 1e-3)
        pl[i] = m.predict(Xte[i:i + 1])[0]
    return np.exp(ote) * (a_local * pl + (1 - a_local) * pg)


if __name__ == "__main__":
    t0 = time.time()
    bG, alpha, llG = nb_global(Xd, y, off)
    aicG = -2 * llG + 2 * (p + 2)
    print(f"NB global: alpha={alpha:.3f} AIC={aicG:.1f} ({time.time()-t0:.0f}s)", flush=True)
    # banda GWNBR por AICc
    grid = [60, 80, 100, 130, 160, 200, 260, 340, 450, 600]
    sel = []
    for k in grid:
        _, _, ll, trS, aicc = gwnbr_fit(C, Xd, y, off, alpha, k); sel.append({"k": k, "ll": ll, "trS": trS, "AICc": aicc})
        print(f"  GWNBR k={k}: AICc={aicc:.1f} tr(S)={trS:.1f}", flush=True)
    sel = pd.DataFrame(sel); sel.to_csv(os.path.join(D, "gwnbr_banda.csv"), index=False)
    k_best = int(sel.loc[sel.AICc.idxmin(), "k"])
    B, mu, ll, trS, aicc = gwnbr_fit(C, Xd, y, off, alpha, k_best)
    loc = pd.DataFrame(B, columns=["intercepto"] + VARS)
    pd.concat([ag[["gx", "gy", "x", "y", "bioma", "focos"]], loc.add_prefix("b_")], axis=1).to_csv(os.path.join(D, "gwnbr_coef_locais.csv"), index=False)
    res = {"alpha_global": float(alpha), "AIC_NB_global": float(aicG), "k_GWNBR": k_best, "AICc_GWNBR": float(aicc), "trS": float(trS)}
    json.dump(res, open(os.path.join(D, "gwnbr_ajuste.json"), "w"), indent=1); print(res, flush=True)
    print(loc.groupby(ag.bioma).median().T.round(3).to_string(), flush=True)

    # validação por blocos de 2°
    rows = []; aloc_grid = [0.25, 0.5, 0.75]
    for rep in range(3):
        for fi, (tr, te) in enumerate(folds_spatial(ag.x.values, ag.y.values, 2.0, 5, seed=rep)):
            bnb, a_tr, _ = nb_global(Xd[tr], y[tr], off[tr])
            preds = {"NB global": np.exp(Xd[te] @ bnb + off[te]),
                     "GWPR": poisson_like_predict(C[tr], Xd[tr], y[tr], off[tr], C[te], Xd[te], off[te], 151),
                     "GWNBR": gwnbr_predict(C[tr], Xd[tr], y[tr], off[tr], C[te], Xd[te], off[te], a_tr, k_best)}
            rfg = rf_fit(X[tr], y[tr], off[tr], seed=rep); preds["RF global"] = rfg.predict(X[te]) * np.exp(off[te])
            for a in aloc_grid:
                preds[f"GWRF (a={a})"] = gwrf_predict(C[tr], X[tr], y[tr], off[tr], C[te], X[te], off[te], 150, a, rfg, seed=rep)
            gb = lgb.LGBMRegressor(objective="poisson", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1, n_jobs=2)
            gb.fit(X[tr], y[tr], init_score=off[tr]); preds["LightGBM"] = np.exp(gb.predict(X[te], raw_score=True) + off[te])
            for nm, pr in preds.items():
                m = metrics(y[te], np.clip(pr, 1e-6, None)); m.update(modelo=nm, rep=rep, fold=fi); rows.append(m)
            print(f"rep {rep} fold {fi} ok ({time.time()-t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "gw_avancado_cv_folds.csv"), index=False)
    S = R.groupby("modelo")[["D2", "Spearman", "MAE"]].agg(["median", "mean", "std"]).round(3)
    S.to_csv(os.path.join(D, "gw_avancado_cv_resumo.csv")); print(S.to_string())
