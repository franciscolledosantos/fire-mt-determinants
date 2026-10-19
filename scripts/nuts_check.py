"""Checagem da inferência variacional: ZINB hierárquico (com efeito ano×bioma e bloco 0,5°) por NUTS
numa subamostra estratificada de 20 mil células-ano, comparada ao SVI na MESMA subamostra."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro
from numpyro.infer import MCMC, NUTS
import glmm_zinb
from glmm_zinb import model, ajustar
from cv_modelos import preparar, LU, CLIM, D

pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet")); pan, _, _ = preparar(pan)
rng = np.random.default_rng(7)
idx = np.concatenate([rng.choice(np.where(pan.bioma == b)[0], n, replace=False) for b, n in [("Amazonia", 8000), ("Cerrado", 8000), ("Pantanal", 4000)]])
pan = pan.iloc[idx].reset_index(drop=True)
FX = CLIM + ["conv"] + LU
X = pan[FX].values.astype(np.float32); Z = pan[["elev", "slope", "lroad", "lpop"]].values.astype(np.float32); y = pan.focos.values.astype(float)
bios = sorted(pan.bioma.unique()); bio = pan.bioma.map({b: i for i, b in enumerate(bios)}).values
anos = sorted(pan.year.unique()); yb = pan.year.map({a: i for i, a in enumerate(anos)}).values * len(bios) + bio
blk = pd.factorize((np.floor(pan.x / 0.5).astype(int)).astype(str) + "_" + (np.floor(pan.y / 0.5).astype(int)).astype(str))[0]
nb, nyb, nblk, N = len(bios), int(yb.max()) + 1, int(blk.max()) + 1, len(y)
t0 = time.time()
mc = MCMC(NUTS(model, target_accept_prob=0.85, max_tree_depth=8), num_warmup=400, num_samples=400, num_chains=1, progress_bar=False)
mc.run(jax.random.PRNGKey(0), jnp.asarray(X), jnp.asarray(Z), jnp.asarray(bio), jnp.asarray(yb), jnp.asarray(blk), nb, nyb, nblk, N, y=jnp.asarray(y))
s = {k: np.asarray(v) for k, v in mc.get_samples().items()}
print(f"NUTS {time.time()-t0:.0f}s; divergências: {int(np.asarray(mc.get_extra_fields()['diverging']).sum())}", flush=True)
post_v, _ = ajustar(X, Z, bio, yb, blk, y, steps=8000, batch=N)
rows = []
for nome, ps in [("NUTS", s), ("SVI", {k: np.asarray(v) for k, v in post_v.items()})]:
    beta = ps["beta"]; delta = ps["delta"] * ps["tau_d"][:, None, :]
    for j, f in enumerate(FX):
        for b, nm in enumerate(bios):
            e = beta[:, j] + delta[:, b, j]; rows.append({"metodo": nome, "variavel": f, "bioma": nm, "media": e.mean(), "dp": e.std()})
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "nuts_vs_svi.csv"), index=False)
P = R.pivot_table(index=["variavel", "bioma"], columns="metodo", values=["media", "dp"]).round(3)
print(P.to_string()); print("corr médias:", np.corrcoef(P[("media", "NUTS")], P[("media", "SVI")])[0, 1].round(3), " razão mediana dp SVI/NUTS:", np.median(P[("dp", "SVI")] / P[("dp", "NUTS")]).round(2))
