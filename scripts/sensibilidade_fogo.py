"""Sensibilidade com a área queimada do MapBiomas Fogo C5.
1) concordância focos × área queimada; 2) GLMs com interações clima × bioma e conversão × bioma:
   NB nos focos (base), NB nos focos + área queimada em t−1 e t−2 (endogeneidade da conversão),
   Tweedie na área queimada (ha), com e sem defasagens; 3) previsão de 2025 da área queimada."""
import os, sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from glum import GeneralizedLinearRegressor, TweedieDistribution
from cv_modelos import preparar, LU, CLIM, STAT, D
from cv_lib import metrics

pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025_fogo.parquet"))
pan, _, Xb = preparar(pan)
pan["lq1"] = np.log1p(100 * pan.queim_t1.fillna(0)); pan["lq2"] = np.log1p(100 * pan.queim_t2.fillna(0))
BIOS = ["Amazonia", "Cerrado", "Pantanal"]
out = {}
# 1) concordância
c = {}
for b in BIOS:
    s = pan[(pan.bioma == b) & (pan.year <= 2024)]
    tot = s.groupby("year")[["focos", "queim_ha"]].sum()
    c[b] = {"spearman_celula_ano": round(spearmanr(s.focos, s.queim_ha)[0], 3),
            "pearson_log_celula_ano": round(pearsonr(np.log1p(s.focos), np.log1p(s.queim_ha))[0], 3),
            "pearson_totais_anuais": round(pearsonr(tot.focos, tot.queim_ha)[0], 3),
            "frac_celulas_ano_com_queima": round((s.queim_ha > 0).mean(), 3), "frac_com_foco": round((s.focos > 0).mean(), 3)}
out["concordancia"] = c; print(json.dumps(c, indent=1), flush=True)

def desenho(df, lags=False):
    X = df[LU + CLIM + STAT + ["conv"]].copy()
    for b in ["Cerrado", "Pantanal"]:
        d = (df.bioma == b).astype(float).values; X[b] = d
        for v in CLIM + ["conv"]: X[f"{v}:{b}"] = X[v].values * d
    if lags: X["lq1"] = df.lq1.values; X["lq2"] = df.lq2.values
    return X

def rr(m, X):
    co = dict(zip(X.columns, m.coef_)); r = {}
    for b in BIOS:
        g = lambda v: co[v] + (co.get(f"{v}:{b}", 0.0) if b != "Amazonia" else 0.0)
        r[b] = {"jjas_-1DP": round(np.exp(-g("p_jjas_anom")), 3), "chuvosa_-1DP": round(np.exp(-g("p_chuvosa_pre_anom")), 3),
                "cdd_+1DP": round(np.exp(g("cdd_anom")), 3), "conv_0a10pp": round(np.exp(g("conv") * np.log(11)), 2)}
    if "lq1" in co: r["lq1_por_unidade_log"] = round(co["lq1"], 3); r["lq2_por_unidade_log"] = round(co["lq2"], 3)
    return r

tr = pan[pan.year <= 2024].reset_index(drop=True)
y_f = tr.focos.values.astype(float); y_q = tr.queim_ha.values
res = {}
for nome, fam, y, lags in [("NB focos (base)", "negative.binomial", y_f, False), ("NB focos + queimada t−1, t−2", "negative.binomial", y_f, True),
                           ("Tweedie área queimada", TweedieDistribution(1.5), y_q, False), ("Tweedie área queimada + queimada t−1, t−2", TweedieDistribution(1.5), y_q, True)]:
    X = desenho(tr, lags)
    m = GeneralizedLinearRegressor(family=fam, alpha=1e-4, l1_ratio=0).fit(X.values, y)
    m.feature_names_in_ = None; res[nome] = rr(m, X); print(nome, json.dumps(res[nome]), flush=True)
out["efeitos"] = res

# 3) 2025 na área queimada: treino 2010–2024
te = pan[pan.year == 2025].reset_index(drop=True)
Xtr, Xte = desenho(tr), desenho(te)
glm = GeneralizedLinearRegressor(family=TweedieDistribution(1.5), alpha=1e-4, l1_ratio=0).fit(Xtr.values, y_q)
gb = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, num_leaves=63, min_child_samples=50, learning_rate=0.05,
                       n_estimators=400, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=1)
gb.fit(Xb[pan.year.values <= 2024], y_q)
p25 = {"GLM Tweedie": glm.predict(Xte.values), "LightGBM Tweedie": gb.predict(Xb[pan.year.values == 2025])}
media = tr.groupby(["x", "y"]).queim_ha.mean(); p25["Média histórica da célula"] = te.set_index(["x", "y"]).index.map(media).values.astype(float)
t25 = {}
for k, p in p25.items():
    d = {"obs_total_mil_ha": round(te.queim_ha.sum() / 1e3, 1), "pred_total_mil_ha": round(p.sum() / 1e3, 1),
         "spearman": round(spearmanr(te.queim_ha, p)[0], 3)}
    for b in BIOS:
        k_ = (te.bioma == b).values; d[f"razao_pred_obs_{b}"] = round(p[k_].sum() / te.queim_ha.values[k_].sum(), 2)
    t25[k] = d
out["teste_2025_area_queimada"] = t25; print(json.dumps(t25, indent=1))
tot = pan.groupby(["year", "bioma"]).agg(focos=("focos", "sum"), queim_mil_ha=("queim_ha", lambda s: s.sum() / 1e3)).round(1)
tot.to_csv(os.path.join(D, "fogo_totais_bioma_ano.csv"))
json.dump(out, open(os.path.join(D, "sensibilidade_fogo.json"), "w"), indent=1, ensure_ascii=False)
