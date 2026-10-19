"""Revisão 17/10/2026 — pedido 4 do revisor.
(a) Teste prospectivo de 2025 para focos com a área queimada de t−1 e t−2 (fração da célula,
    MapBiomas Fogo C5) como covariáveis, com e sem focos de t−1; NB-GLM e LightGBM (Poisson).
(b) Anomalias climáticas médias por bioma em 2025 (e em 2020 e 2024 para comparação)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from cv_modelos import preparar, fp_glm, make_gbm, D
from cv_lib import metrics

OUT = sys.argv[1] if len(sys.argv) > 1 else D
os.makedirs(OUT, exist_ok=True)
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025_fogo.parquet"))
pan, Xg, Xb = preparar(pan); y = pan.focos.values.astype(float)
tr = (pan.year <= 2024).values; te = (pan.year == 2025).values; bios = pan.bioma.values
lq1 = np.log1p(100 * pan.queim_t1.fillna(0).values); lq2 = np.log1p(100 * pan.queim_t2.fillna(0).values)
lag = pan[["x", "y", "year", "focos"]].copy(); lag["year"] += 1
fa = pan.merge(lag.rename(columns={"focos": "focos_ant"}), on=["x", "y", "year"], how="left").focos_ant
lf1 = np.log1p(fa.fillna(0).values); ok_f = fa.notna().values
gbm_default = dict(num_leaves=63, min_child_samples=50, learning_rate=0.05, n_estimators=400, colsample_bytree=0.8, subsample=0.8, reg_lambda=0)
hp = json.load(open(os.path.join(D, "hiperparametros_escolhidos.json")))["interno_espacial"]

rows = []
def avaliar(nome, mu, trmask):
    m = metrics(y[te], np.clip(mu, 1e-6, None), y[trmask].mean()); m["modelo"] = nome
    for b in ["Amazonia", "Cerrado", "Pantanal"]:
        k = bios[te] == b; m[f"obs_{b}"] = y[te][k].sum(); m[f"pred_{b}"] = float(mu[k].sum()); m[f"razao_{b}"] = round(m[f"pred_{b}"] / m[f"obs_{b}"], 2)
    m["razao_total"] = round(m["soma_pred"] / m["soma_obs"], 2); rows.append(m); print(nome, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}, flush=True)

variantes = {"sem defasagem": (None, tr), "+ área queimada t−1, t−2": (["lq1", "lq2"], tr),
             "+ focos t−1": (["lf1"], tr & ok_f), "+ área queimada t−1, t−2 + focos t−1": (["lq1", "lq2", "lf1"], tr & ok_f)}
extras = {"lq1": lq1, "lq2": lq2, "lf1": lf1}
for vn, (cols, trm) in variantes.items():
    Xg2, Xb2 = Xg.copy(), Xb.copy()
    for c in (cols or []): Xg2[c] = extras[c]; Xb2[c] = extras[c]
    avaliar(f"NB-GLM, {vn}", fp_glm(Xg2[trm], y[trm], Xg2[te]), trm)
    avaliar(f"LightGBM (padrão), {vn}", make_gbm(gbm_default)(Xb2[trm], y[trm], Xb2[te]), trm)
    avaliar(f"LightGBM (ajustado), {vn}", make_gbm(hp)(Xb2[trm], y[trm], Xb2[te]), trm)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "teste_2025_defasagens.csv"), index=False)
print(R[["modelo", "D2", "AUC_ocorr", "Spearman", "razao_total", "razao_Amazonia", "razao_Cerrado", "razao_Pantanal"]].round(3).to_string())

# (b) anomalias por bioma e ano
A = pan.groupby(["year", "bioma"])[["p_jjas_anom", "p_chuvosa_pre_anom", "cdd_anom"]].mean().round(2)
A.to_csv(os.path.join(OUT, "anomalias_bioma_ano.csv"))
print(A.loc[[2010, 2019, 2020, 2024, 2025]].to_string())
# área queimada 2025 × 2024 por bioma e totais
T = pan.groupby(["year", "bioma"]).agg(focos=("focos", "sum"), queim_mil_ha=("queim_ha", lambda s: s.sum() / 1e3)).round(1)
T.to_csv(os.path.join(OUT, "focos_area_bioma_ano.csv")); print(T.loc[[2024, 2025]].to_string())
