"""Teste prospectivo: treino 2010–2024, previsão de 2025 (ano fora de toda a modelagem anterior).
Requer MapBiomas 2024 (uso t−1) e CHIRPS/focos 2025. Climatologia das anomalias continua 2010–2024 (sem vazamento)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from montar_painel import montar
from cv_modelos import preparar, fp_glm, make_gbm, D
from cv_lib import metrics

arq = os.path.join(D, "painel_MT_2010_2025.parquet")
if not os.path.exists(arq):
    montar(anos=range(2010, 2026)).to_parquet(arq)
pan, Xg, Xb = preparar(pd.read_parquet(arq)); y = pan.focos.values.astype(float)
tr = (pan.year <= 2024).values; te = (pan.year == 2025).values
print("linhas treino/teste:", tr.sum(), te.sum(), "focos 2025:", y[te].sum(), flush=True)
hp = json.load(open(os.path.join(D, "hiperparametros_escolhidos.json")))
gbm_default = dict(num_leaves=63, min_child_samples=50, learning_rate=0.05, n_estimators=400, colsample_bytree=0.8, subsample=0.8, reg_lambda=0)
preds = {"NB-GLM": fp_glm(Xg[tr], y[tr], Xg[te]),
         "LightGBM (padrão)": make_gbm(gbm_default)(Xb[tr], y[tr], Xb[te]),
         "LightGBM (ajustado, CV interna espacial)": make_gbm(hp["interno_espacial"])(Xb[tr], y[tr], Xb[te])}
media_cel = pan[tr].groupby(["x", "y"]).focos.mean()
preds["Média histórica da célula (referência ingênua)"] = pan[te].set_index(["x", "y"]).index.map(media_cel).values.astype(float)
rows = []; bios = pan.bioma.values[te]
for nm, mu in preds.items():
    m = metrics(y[te], np.clip(mu, 1e-6, None), y[tr].mean()); m["modelo"] = nm
    for b in ["Amazonia", "Cerrado", "Pantanal"]:
        k = bios == b; m[f"obs_{b}"] = y[te][k].sum(); m[f"pred_{b}"] = float(mu[k].sum())
    rows.append(m)
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "teste_2025.csv"), index=False); print(R.round(3).to_string())

# variante: focos do ano anterior na célula (memória de combustível consumido / foco persistente)
lag = pan[["x", "y", "year", "focos"]].copy(); lag["year"] += 1
pan2 = pan.merge(lag.rename(columns={"focos": "focos_ant"}), on=["x", "y", "year"], how="left")
ok = pan2.focos_ant.notna().values
Xb2 = Xb.copy(); Xb2["lfocos_ant"] = np.log1p(pan2.focos_ant.values)
tr2 = tr & ok; rows2 = []
for nm, p in [("LightGBM (padrão) + focos t−1", gbm_default), ("LightGBM (ajustado) + focos t−1", hp["interno_espacial"])]:
    mu = make_gbm(p)(Xb2[tr2], y[tr2], Xb2[te])
    m = metrics(y[te], np.clip(mu, 1e-6, None), y[tr2].mean()); m["modelo"] = nm
    for b in ["Amazonia", "Cerrado", "Pantanal"]:
        k = bios == b; m[f"obs_{b}"] = y[te][k].sum(); m[f"pred_{b}"] = float(mu[k].sum())
    rows2.append(m)
R = pd.concat([R, pd.DataFrame(rows2)]); R.to_csv(os.path.join(D, "teste_2025.csv"), index=False); print(R.round(3).to_string())
