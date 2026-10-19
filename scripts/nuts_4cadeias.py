"""NUTS com 4 cadeias na mesma subamostra estratificada de 20 mil células-ano de nuts_check.py.
Diagnósticos (R-hat, ESS) dos 45 efeitos por bioma e comparação com o SVI (nuts_vs_svi.csv)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import split_gelman_rubin, effective_sample_size
from glmm_zinb import model
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
mc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=8), num_warmup=600, num_samples=500, num_chains=4,
          chain_method="sequential", progress_bar=False)
mc.run(jax.random.PRNGKey(11), jnp.asarray(X), jnp.asarray(Z), jnp.asarray(bio), jnp.asarray(yb), jnp.asarray(blk), nb, nyb, nblk, N,
       y=jnp.asarray(y), extra_fields=("diverging",))
s = {k: np.asarray(v) for k, v in mc.get_samples(group_by_chain=True).items()}
div = int(np.asarray(mc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
eff = s["beta"][:, :, None, :] + s["delta"] * s["tau_d"][:, :, None, :]       # (cadeia, amostra, bioma, var)
rows = []
for j, f in enumerate(FX):
    for b, nm in enumerate(bios):
        e = eff[:, :, b, j]
        rows.append({"variavel": f, "bioma": nm, "media": e.mean(), "dp": e.std(), "q025": np.quantile(e, .025), "q975": np.quantile(e, .975),
                     "rhat": float(split_gelman_rubin(e)), "ess": float(effective_sample_size(e))})
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "nuts_4cadeias.csv"), index=False)
hyp = {k: (float(split_gelman_rubin(s[k])), float(effective_sample_size(s[k]))) for k in ["s_u", "s_s", "g0"]}
V = pd.read_csv(os.path.join(D, "nuts_vs_svi.csv")); V = V[V.metodo == "SVI"].rename(columns={"media": "media_svi", "dp": "dp_svi"})
M = R.merge(V[["variavel", "bioma", "media_svi", "dp_svi"]], on=["variavel", "bioma"])
res = {"tempo_s": round(time.time() - t0), "divergencias": div, "n_amostras": int(eff.shape[0] * eff.shape[1]),
       "rhat_max": float(R.rhat.max()), "ess_min": float(R.ess.min()), "ess_mediana": float(R.ess.median()),
       "corr_medias_svi": float(np.corrcoef(M.media, M.media_svi)[0, 1]), "razao_mediana_dp_svi_nuts": float(np.median(M.dp_svi / M.dp)),
       "hiperparametros_rhat_ess": hyp}
json.dump(res, open(os.path.join(D, "nuts_4cadeias_resumo.json"), "w"), indent=1)
print(json.dumps(res, indent=1)); print(M.assign(razao=M.dp_svi / M.dp).round(3).to_string())
