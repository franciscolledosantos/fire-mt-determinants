import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, jax
import glmm_zinb
from glmm_zinb import ajustar
from cv_modelos import preparar, LU, CLIM, STAT

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados")
arq = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "painel_MT_2010_2024.parquet")
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
tag = sys.argv[3] if len(sys.argv) > 3 else "completo"
glmm_zinb.USE_YEAR = (len(sys.argv) <= 4 or sys.argv[4] != "sem_ano")

pan = pd.read_parquet(arq)
pan, _, _ = preparar(pan)
y = pan.focos.values.astype(float)
FX = CLIM + ["conv"] + LU
X = pan[FX].values.astype(np.float32)
Z = pan[["elev", "slope", "lroad", "lpop"]].values.astype(np.float32)
bios = sorted(pan.bioma.unique()); bio = pan.bioma.map({b: i for i, b in enumerate(bios)}).values
anos = sorted(pan.year.unique())
yb = (pan.year.map({a: i for i, a in enumerate(anos)}).values * len(bios) + bio)
bx = np.floor(pan.x / 0.5).astype(int); by = np.floor(pan.y / 0.5).astype(int)
blk = pd.factorize(bx.astype(str) + "_" + by.astype(str))[0]
print(f"n={len(y)} p={X.shape[1]} biomas={bios} anos={len(anos)} blocos={blk.max()+1}", flush=True)

post, res = ajustar(X, Z, bio, yb, blk, y, steps=steps)
beta = np.asarray(post["beta"]); delta = np.asarray(post["delta"]) * np.asarray(post["tau_d"])[:, None, :]
rows = []
for j, f in enumerate(FX):
    for b, nm in enumerate(bios):
        e = beta[:, j] + delta[:, b, j]
        rows.append({"variavel": f, "bioma": nm, "media": e.mean(), "q025": np.quantile(e, .025), "q975": np.quantile(e, .975),
                     "RR_media": np.exp(e).mean()})
eff = pd.DataFrame(rows)
eff.to_csv(os.path.join(D, f"glmm_efeitos_{tag}.csv"), index=False)
g = np.asarray(post["gamma"]); gz = pd.DataFrame({"var": ["elev", "slope", "lroad", "lpop"], "media": g.mean(0), "q025": np.quantile(g, .025, 0), "q975": np.quantile(g, .975, 0)})
gz.to_csv(os.path.join(D, f"glmm_zeros_{tag}.csv"), index=False)
var = {k: float(np.asarray(post[k]).mean()) for k in ["s_u", "s_s"]}; var["phi"] = np.asarray(post["phi"]).mean(0).round(3).tolist()
json.dump(var, open(os.path.join(D, f"glmm_variancias_{tag}.json"), "w"), indent=1)
np.save(os.path.join(D, f"glmm_elbo_{tag}.npy"), np.asarray(res.losses))
print(eff.pivot(index="variavel", columns="bioma", values="media").round(3).to_string())
print(gz.round(3).to_string()); print(var)
