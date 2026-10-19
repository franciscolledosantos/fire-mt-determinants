"""IC de 95% (sanduíche robusto do glum) para as razões da Tabela 4."""
import os, sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from glum import GeneralizedLinearRegressor, TweedieDistribution
from cv_modelos import preparar, LU, CLIM, STAT, D
exec(open(os.path.join(os.path.dirname(__file__), "sensibilidade_fogo.py")).read().split("def rr(")[0].split("pan = pd.read_parquet")[0])
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025_fogo.parquet")); pan, _, _ = preparar(pan)
pan["lq1"] = np.log1p(100 * pan.queim_t1.fillna(0)); pan["lq2"] = np.log1p(100 * pan.queim_t2.fillna(0))
src = open(os.path.join(os.path.dirname(__file__), "sensibilidade_fogo.py")).read()
exec(src[src.index("def desenho("):src.index("def rr(")])
tr = pan[pan.year <= 2024].reset_index(drop=True)
s = tr[tr.bioma == "Cerrado"].groupby("year")[["focos", "queim_ha"]].sum(); print("r Cerrado:", round(pearsonr(s.focos, s.queim_ha)[0], 4))
EF = {"p_jjas_anom": -1, "p_chuvosa_pre_anom": -1, "cdd_anom": 1, "conv": np.log(11)}
out = {}
for nome, fam, y, lags in [("focos", "negative.binomial", tr.focos.values.astype(float), False), ("focos+lag", "negative.binomial", tr.focos.values.astype(float), True),
                           ("area", TweedieDistribution(1.5), tr.queim_ha.values, False), ("area+lag", TweedieDistribution(1.5), tr.queim_ha.values, True)]:
    X = desenho(tr, lags); cols = list(X.columns)
    m = GeneralizedLinearRegressor(family=fam, alpha=0, fit_intercept=True).fit(X.values, y)
    V = np.asarray(m.covariance_matrix(X.values, y, robust=True)); b = np.r_[m.intercept_, m.coef_]
    r = {}
    for v, k in EF.items():
        for bio in ["Amazonia", "Cerrado", "Pantanal"]:
            w = np.zeros(len(b)); w[1 + cols.index(v)] = 1
            if bio != "Amazonia": w[1 + cols.index(f"{v}:{bio}")] = 1
            est, se = w @ b, np.sqrt(w @ V @ w)
            lo, hi = sorted([np.exp(k * (est - 1.96 * se)), np.exp(k * (est + 1.96 * se))])
            r[f"{v}|{bio}"] = [round(np.exp(k * est), 2), round(lo, 2), round(hi, 2)]
    out[nome] = r; print(nome, r, flush=True)
json.dump(out, open(os.path.join(D, "sensibilidade_fogo_ic.json"), "w"), indent=1)
