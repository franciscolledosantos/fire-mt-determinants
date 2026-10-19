"""2025 na área queimada com as queimadas de t−1 e t−2 como covariáveis (combustível consumido)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr
from cv_modelos import preparar, D
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025_fogo.parquet")); pan, _, Xb = preparar(pan)
Xb = Xb.copy(); Xb["lq1"] = np.log1p(100 * pan.queim_t1.fillna(0)).values; Xb["lq2"] = np.log1p(100 * pan.queim_t2.fillna(0)).values
tr = (pan.year <= 2024).values; te = (pan.year == 2025).values; y = pan.queim_ha.values
gb = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, num_leaves=63, min_child_samples=50, learning_rate=0.05,
                       n_estimators=400, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=1).fit(Xb[tr], y[tr])
p = gb.predict(Xb[te]); b = pan.bioma.values[te]
r = {"pred_total_mil_ha": round(p.sum() / 1e3, 1), "obs_total_mil_ha": round(y[te].sum() / 1e3, 1), "spearman": round(spearmanr(y[te], p)[0], 3)}
for bb in ["Amazonia", "Cerrado", "Pantanal"]: r[f"razao_{bb}"] = round(p[b == bb].sum() / y[te][b == bb].sum(), 2)
print(json.dumps(r)); json.dump(r, open(os.path.join(D, "teste_2025_fogo_lag.json"), "w"))
