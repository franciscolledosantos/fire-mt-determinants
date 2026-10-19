"""Revalida só o NB-GLM (com conversão em log) nos mesmos esquemas e junta aos resultados do LightGBM."""
import os, sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from cv_modelos import preparar, fp_glm, D
from cv_lib import folds_random, folds_spatial, folds_temporal, metrics
pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet")); pan, Xg, Xb = preparar(pan); y = pan.focos.values
esq = {"aleatória (5 folds)": folds_random(len(y), 5), "blocos espaciais 1° (5 folds)": folds_spatial(pan.x.values, pan.y.values, 1.0, 5),
       "blocos espaciais 2° (5 folds)": folds_spatial(pan.x.values, pan.y.values, 2.0, 5), "temporal (teste a partir de T_TESTE)": folds_temporal(pan.year.values, 2020)}
out = []
for nome, fo in esq.items():
    for i, (tr, te) in enumerate(fo):
        mu = fp_glm(Xg.iloc[tr], y[tr], Xg.iloc[te]); m = metrics(y[te], mu, y[tr].mean()); m.update(modelo="NB-GLM", esquema=nome, fold=i); out.append(m)
R0 = pd.read_csv(os.path.join(D, "cv_resultados_folds.csv")); R0 = R0[R0.modelo != "NB-GLM"]
R = pd.concat([R0, pd.DataFrame(out)]); R.to_csv(os.path.join(D, "cv_resultados_folds_final.csv"), index=False)
S = R.groupby(["modelo", "esquema"])[["D2", "AUC_ocorr", "Spearman", "MAE"]].agg(["mean", "std"]).round(3); S.to_csv(os.path.join(D, "cv_resumo_final.csv"))
print(S.to_string()); print(pd.DataFrame(out)[["esquema", "fold", "D2", "soma_obs", "soma_pred"]].round(2).to_string())
