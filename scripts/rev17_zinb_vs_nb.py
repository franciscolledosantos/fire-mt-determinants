"""Revisão 17/10/2026 — pedido 2 do revisor: justificar a inflação de zeros por critério.
NB-GLM × ZINB-GLM (sem efeitos aleatórios), mesmas covariáveis do NB-GLM de cv_modelos.py
(clima × bioma, conversão, uso do solo, estáticas, bioma); inflação de zeros com logit em
elev, slope, log(dist. estradas), log(pop.) — os mesmos Z do ZINB hierárquico.
Saídas: AIC/BIC, teste de Vuong, fração de zeros prevista, D² em 5 folds de blocos de 1° e no
teste temporal (2020–24); média histórica da célula (2010–19) como referência no teste temporal.
Máxima verossimilhança por L-BFGS com gradiente analítico (numpy, float64; leve em memória)."""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.special import gammaln, digamma, expit
from scipy.optimize import minimize
from scipy import stats
from cv_modelos import preparar, D
from cv_lib import folds_spatial, folds_temporal, metrics

OUT = sys.argv[1] if len(sys.argv) > 1 else D
os.makedirs(OUT, exist_ok=True)
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet"))
pan, Xg, Xb = preparar(pan); y = pan.focos.values.astype(float)
X = np.column_stack([np.ones(len(y)), Xg.values.astype(float)])
Z = np.column_stack([np.ones(len(y)), pan[["elev", "slope", "lroad", "lpop"]].values.astype(float)])
p, q = X.shape[1], Z.shape[1]; L2 = 1e-4
print("n", len(y), "p", p, "q", q, flush=True)


def nb_parts(b, la, X, y):
    r = np.exp(-la); eta = np.clip(X @ b, -20, 15); mu = np.exp(eta)
    lnb = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))
    # d lnb / d eta, d lnb / d la
    dmu = (y - mu) * r / (r + mu)
    dr = digamma(y + r) - digamma(r) + np.log(r / (r + mu)) + 1 - (y + r) / (r + mu)
    dla = -r * dr
    return r, mu, lnb, dmu, dla


def negll_nb(t, X, y):
    b, la = t[:p], t[-1]
    r, mu, lnb, dmu, dla = nb_parts(b, la, X, y)
    f = -lnb.sum() + L2 * np.sum(b[1:] ** 2)
    g = np.empty_like(t); g[:p] = -(X.T @ dmu); g[1:p] += 2 * L2 * b[1:]; g[-1] = -dla.sum()
    return f, g


def negll_zinb(t, X, Z, y):
    b, gm, la = t[:p], t[p:p + q], t[-1]
    r, mu, lnb, dmu, dla = nb_parts(b, la, X, y)
    zeta = Z @ gm; pi = expit(zeta); lp = -np.logaddexp(0, -zeta); l1p = -np.logaddexp(0, zeta)  # log pi, log(1-pi)
    z0 = y == 0
    lp0 = r * np.log(r / (r + mu))                    # log P_NB(0)
    la0 = np.logaddexp(lp, l1p + lp0)                 # log(pi + (1-pi) P0)
    ll = np.where(z0, la0, l1p + lnb)
    # gradients
    w = np.exp(l1p + lp0 - la0)                       # peso de (1-pi)P0 no zero
    dP0_deta = -r * mu / (r + mu)                     # d log P0 / d eta
    dP0_dla = -r * (np.log(r / (r + mu)) + 1 - r / (r + mu))  # d log P0 / d la  (r = e^-la)
    geta = np.where(z0, w * dP0_deta, dmu)
    gla = np.where(z0, w * dP0_dla, dla)
    gzeta = np.where(z0, pi * (1 - pi) * (1 - np.exp(lp0)) * np.exp(-la0), -pi)  # d/dzeta of la0 or l1p
    f = -ll.sum() + L2 * (np.sum(b[1:] ** 2) + np.sum(gm[1:] ** 2))
    g = np.empty_like(t); g[:p] = -(X.T @ geta); g[1:p] += 2 * L2 * b[1:]
    g[p:p + q] = -(Z.T @ gzeta); g[p + 1:p + q] += 2 * L2 * gm[1:]; g[-1] = -gla.sum()
    return f, g, ll


def fit(kind, X, Z, y, x0=None):
    if kind == "nb":
        fun = lambda t: negll_nb(t, X, y); n_par = p + 1
    else:
        fun = lambda t: negll_zinb(t, X, Z, y)[:2]; n_par = p + q + 1
    t0 = np.zeros(n_par) if x0 is None else x0.copy()
    if x0 is None:
        t0[0] = np.log(y.mean() + 1e-6); t0[-1] = np.log(2.0)
        if kind == "zinb": t0[p] = -1.0
    r = minimize(fun, t0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=4000, ftol=1e-12, gtol=1e-6))
    return r.x, r


def predict(kind, theta, X, Z):
    mu = np.exp(np.clip(X @ theta[:p], -20, 15))
    if kind == "nb": return mu
    return (1 - expit(Z @ theta[p:p + q])) * mu


def zeros_pred(kind, theta, X, Z):
    r = np.exp(-theta[-1]); mu = np.exp(np.clip(X @ theta[:p], -20, 15)); p0 = (r / (r + mu)) ** r
    if kind == "nb": return p0.mean()
    pi = expit(Z @ theta[p:p + q]); return (pi + (1 - pi) * p0).mean()


# checagem numérica do gradiente numa subamostra
sub = np.random.default_rng(0).choice(len(y), 5000, replace=False)
t = np.random.default_rng(1).normal(0, 0.1, p + q + 1); t[-1] = 0.3
f0, g0, _ = negll_zinb(t, X[sub], Z[sub], y[sub]); eps = 1e-6; gn = np.zeros_like(t)
for j in range(len(t)):
    tt = t.copy(); tt[j] += eps; gn[j] = (negll_zinb(tt, X[sub], Z[sub], y[sub])[0] - f0) / eps
print("gradiente ZINB: erro máx. relativo", float(np.max(np.abs(gn - g0) / (np.abs(g0) + 1))), flush=True)

res = {}; th = {}; ll = {}
for kind in ["nb", "zinb"]:
    t0 = time.time(); th[kind], r = fit(kind, X, Z, y)
    if kind == "nb":
        ll[kind] = nb_parts(th[kind][:p], th[kind][-1], X, y)[2]
    else:
        ll[kind] = negll_zinb(th[kind], X, Z, y)[2]
    k = len(th[kind]); Lk = float(ll[kind].sum())
    res[kind] = {"logLik": round(Lk, 1), "k": k, "AIC": round(-2 * Lk + 2 * k, 1), "BIC": round(-2 * Lk + k * np.log(len(y)), 1),
                 "alpha": round(float(np.exp(th[kind][-1])), 4), "zeros_prev": round(float(zeros_pred(kind, th[kind], X, Z)), 4),
                 "convergiu": bool(r.success), "iteracoes": int(r.nit), "segundos": round(time.time() - t0)}
    print(kind, res[kind], flush=True)
m = ll["zinb"] - ll["nb"]; n = len(m)
V = np.sqrt(n) * m.mean() / m.std(ddof=1)
res["vuong_zinb_vs_nb"] = {"estatistica": round(float(V), 2), "p_unilateral": float(stats.norm.sf(V)),
                           "nota": "positivo favorece o ZINB; ignora a dependência espacial (otimista)"}
res["zeros_obs"] = round(float((y == 0).mean()), 4)
res["gamma_zero_inflation"] = dict(zip(["intercept", "elev", "slope", "lroad", "lpop"], np.round(th["zinb"][p:p + q], 3).tolist()))
res["pi_estrutural_media"] = round(float(expit(Z @ th["zinb"][p:p + q]).mean()), 4)
print("Vuong", res["vuong_zinb_vs_nb"], flush=True)
json.dump(res, open(os.path.join(OUT, "zinb_vs_nb_ajuste.json"), "w"), indent=1, ensure_ascii=False)

rows = []
esq = {"blocos espaciais 1° (5 folds)": folds_spatial(pan.x.values, pan.y.values, 1.0, 5),
       "temporal (teste 2020–24)": folds_temporal(pan.year.values, 2020)}
for nome, fo in esq.items():
    for i, (tr, te) in enumerate(fo):
        for kind in ["nb", "zinb"]:
            t0 = time.time(); t, _ = fit(kind, X[tr], Z[tr], y[tr], x0=th[kind])
            mu = predict(kind, t, X[te], Z[te])
            mm = metrics(y[te], np.clip(mu, 1e-9, None), y[tr].mean()); mm.update(modelo={"nb": "NB-GLM (ML)", "zinb": "ZINB-GLM (ML)"}[kind], esquema=nome, fold=i, segundos=round(time.time() - t0))
            rows.append(mm); print(nome, i, kind, round(mm["D2"], 3), f"{time.time()-t0:.0f}s", flush=True)
        if nome.startswith("temporal"):
            med = pan.iloc[tr].groupby(["x", "y"]).focos.mean()
            mu = pan.iloc[te].set_index(["x", "y"]).index.map(med).values.astype(float)
            mm = metrics(y[te], np.clip(mu, 1e-9, None), y[tr].mean()); mm.update(modelo="Média histórica da célula (2010–19)", esquema=nome, fold=i, segundos=0)
            rows.append(mm); print(nome, "media historica", round(mm["D2"], 3), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "zinb_vs_nb_folds.csv"), index=False)
R = pd.DataFrame(rows)
S = R.groupby(["modelo", "esquema"])[["D2", "AUC_ocorr", "Spearman", "MAE"]].agg(["mean", "std"]).round(3)
S.to_csv(os.path.join(OUT, "zinb_vs_nb_resumo.csv")); print(S.to_string()); print("ok")
