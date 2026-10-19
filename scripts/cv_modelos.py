"""Comparação preditiva GLM × LightGBM sob validação aleatória, espacial e temporal,
com ajuste de hiperparâmetros do LightGBM por busca aleatória em validação interna
(interna aleatória × interna espacial).
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, lightgbm as lgb
from glum import GeneralizedLinearRegressor
from cv_lib import folds_random, folds_spatial, folds_temporal, metrics

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(W, "resultados")
LU = ["floresta", "savana", "campestre", "alagado", "soja", "outras_temp", "cana", "mosaico", "perene_silvic", "urbano_outros", "agua"]  # pastagem = referência
CLIM = ["cdd_anom", "p_chuvosa_pre_anom", "p_jjas_anom"]
STAT = ["elev", "slope", "lroad", "lpop"]


def preparar(pan):
    pan = pan.dropna(subset=LU + CLIM + ["elev"]).copy()
    pan["lroad"] = np.log1p(pan.roaddist); pan["lpop"] = np.log1p(pan.denspop * 1000)
    for c in ["elev", "slope", "lroad", "lpop"]:
        pan[c] = (pan[c] - pan[c].mean()) / pan[c].std()
    pan["conv"] = np.log1p(pan.conversao.fillna(0) * 100)  # log(1 + p.p. de nativa convertida entre t-2 e t-1)
    bio = pd.get_dummies(pan.bioma).astype(float)
    base = pan[LU + CLIM + STAT + ["conv"]].copy()
    X_glm = pd.concat([base, bio[["Cerrado", "Pantanal"]]], axis=1)
    # interações clima × bioma no GLM (o GBM aprende sozinho)
    for b in ["Cerrado", "Pantanal"]:
        for c in ["cdd_anom", "p_chuvosa_pre_anom", "p_jjas_anom"]:
            X_glm[f"{c}:{b}"] = X_glm[c] * bio[b]
    X_gbm = pd.concat([base, bio], axis=1)
    return pan.reset_index(drop=True), X_glm.reset_index(drop=True), X_gbm.reset_index(drop=True)


def fp_glm(Xtr, ytr, Xte):
    m = GeneralizedLinearRegressor(family="negative.binomial", alpha=1e-4, l1_ratio=0)
    m.fit(Xtr.values, ytr); return m.predict(Xte.values)


def make_gbm(params):
    def fp(Xtr, ytr, Xte):
        m = lgb.LGBMRegressor(objective="poisson", verbose=-1, n_jobs=2, subsample_freq=1, **params)
        m.fit(Xtr, ytr); return m.predict(Xte)
    return fp


ESPACO = dict(num_leaves=[7, 15, 31, 63, 127, 255], min_child_samples=[20, 50, 100, 300, 1000],
              learning_rate=[0.02, 0.05, 0.1], n_estimators=[200, 400, 800],
              colsample_bytree=[0.5, 0.8, 1.0], subsample=[0.6, 0.8, 1.0], reg_lambda=[0, 1, 10])


def tune(X, y, inner_folds, n_iter=20, seed=0):
    rng = np.random.default_rng(seed); best = None; hist = []
    for it in range(n_iter):
        p = {k: v[rng.integers(len(v))] for k, v in ESPACO.items()}
        sc = []
        for tr, te in inner_folds:
            mu = make_gbm(p)(X.iloc[tr], y[tr], X.iloc[te]); sc.append(metrics(y[te], mu, y[tr].mean())["D2"])
        s = float(np.mean(sc)); hist.append({**p, "D2_interno": s})
        if best is None or s > best[1]: best = (p, s)
    return best[0], pd.DataFrame(hist)


if __name__ == "__main__":
    arq = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "painel_MT_2010_2024.parquet")
    TAG = sys.argv[2] if len(sys.argv) > 2 else ""
    T_TESTE = int(sys.argv[3]) if len(sys.argv) > 3 else 2020
    pan = pd.read_parquet(arq)
    pan, Xg, Xb = preparar(pan); y = pan.focos.values
    print("painel", pan.shape, "focos", y.sum(), flush=True)
    esquemas = {
        "aleatória (5 folds)": folds_random(len(y), 5),
        "blocos espaciais 1° (5 folds)": folds_spatial(pan.x.values, pan.y.values, 1.0, 5),
        "blocos espaciais 2° (5 folds)": folds_spatial(pan.x.values, pan.y.values, 2.0, 5),
        "temporal (teste a partir de T_TESTE)": folds_temporal(pan.year.values, T_TESTE),
    }
    out = []
    # 1) GLM e GBM com hiperparâmetros padrão
    gbm_default = dict(num_leaves=63, min_child_samples=50, learning_rate=0.05, n_estimators=400, colsample_bytree=0.8, subsample=0.8, reg_lambda=0)
    for nome, fo in esquemas.items():
        for mod, fp, X in [("NB-GLM", fp_glm, Xg), ("LightGBM (padrão)", make_gbm(gbm_default), Xb)]:
            t0 = time.time()
            for i, (tr, te) in enumerate(fo):
                m = metrics(y[te], fp(X.iloc[tr], y[tr], X.iloc[te]), y[tr].mean()); m.update(modelo=mod, esquema=nome, fold=i); out.append(m)
            print(mod, nome, f"{time.time()-t0:.0f}s", flush=True)
    # 2) ajuste de hiperparâmetros: interno aleatório × interno espacial, avaliado em blocos espaciais 1°
    sub = np.random.default_rng(3).choice(len(y), min(len(y), 120000), replace=False)  # busca em subamostra
    Xs, ys = Xb.iloc[sub].reset_index(drop=True), y[sub]
    p_rand, h1 = tune(Xs, ys, folds_random(len(ys), 3, 1), n_iter=20, seed=1)
    p_spat, h2 = tune(Xs, ys, folds_spatial(pan.x.values[sub], pan.y.values[sub], 1.0, 3, 1), n_iter=20, seed=1)
    h1["interno"] = "aleatório"; h2["interno"] = "espacial"; pd.concat([h1, h2]).to_csv(os.path.join(D, f"busca_hiperparametros{TAG}.csv"), index=False)
    json.dump({"interno_aleatorio": p_rand, "interno_espacial": p_spat}, open(os.path.join(D, f"hiperparametros_escolhidos{TAG}.json"), "w"), indent=1, default=int)
    for lab, p in [("LightGBM (ajustado, CV interna aleatória)", p_rand), ("LightGBM (ajustado, CV interna espacial)", p_spat)]:
        for nome in ["aleatória (5 folds)", "blocos espaciais 1° (5 folds)", "temporal (teste a partir de T_TESTE)"]:
            for i, (tr, te) in enumerate(esquemas[nome]):
                m = metrics(y[te], make_gbm(p)(Xb.iloc[tr], y[tr], Xb.iloc[te]), y[tr].mean()); m.update(modelo=lab, esquema=nome, fold=i); out.append(m)
            print(lab, nome, flush=True)
    R = pd.DataFrame(out); R.to_csv(os.path.join(D, f"cv_resultados_folds{TAG}.csv"), index=False)
    S = R.groupby(["modelo", "esquema"])[["D2", "AUC_ocorr", "Spearman", "MAE"]].agg(["mean", "std"]).round(3)
    S.to_csv(os.path.join(D, f"cv_resumo{TAG}.csv")); print(S.to_string())
    print("escolhidos:", p_rand, p_spat)
