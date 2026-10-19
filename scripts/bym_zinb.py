"""ZINB espacial (BYM: ICAR + iid) no painel 0,05° × ano, com efeito aleatório ano × bioma.

log mu = a[bioma] + u[ano,bioma] + sigma_phi*phi[célula] + sigma_th*theta[célula] + X (beta + delta[bioma])
phi ~ ICAR (vizinhança de rainha entre células de 0,05°), theta ~ N(0,1)
Inferência variacional com minilotes; o termo ICAR usa todas as arestas a cada passo.

Uso: python bym_zinb.py <painel> <passos> <tag> [fold_k_de_5]  — com fold, as células do bloco espacial
de 1° do fold k ficam fora da verossimilhança e são previstas pela suavização espacial (validação honesta).
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer import init_to_uniform
from cv_modelos import preparar, LU, CLIM
from cv_lib import folds_spatial, metrics

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados")


def model(X, Z, bio, yb, cel, E0, E1, nb, nyb, ncel, N, msk=None, y=None):
    p, q = X.shape[1], Z.shape[1]
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([p]))
    tau_d = numpyro.sample("tau_d", dist.HalfNormal(0.5).expand([p]))
    delta = numpyro.sample("delta", dist.Normal(0, 1).expand([nb, p])) * tau_d
    a = numpyro.sample("a", dist.Normal(0, 2).expand([nb]))
    s_u = numpyro.sample("s_u", dist.HalfNormal(1.0)); u = numpyro.sample("u", dist.Normal(0, 1).expand([nyb])) * s_u
    s_phi = numpyro.sample("s_phi", dist.HalfNormal(1.0)); s_th = numpyro.sample("s_th", dist.HalfNormal(1.0))
    phi = numpyro.sample("phi", dist.ImproperUniform(dist.constraints.real, (), (ncel,)))
    msk = jnp.ones(ncel) if msk is None else msk
    numpyro.factor("icar", -0.5 * jnp.sum((phi[E0] - phi[E1]) ** 2) - 0.5 * (jnp.sum(phi * msk) / (0.001 * jnp.sum(msk))) ** 2)
    th = numpyro.sample("th", dist.Normal(0, 1).expand([ncel]))
    gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([q])); g0 = numpyro.sample("g0", dist.Normal(-1, 2))
    phid = numpyro.sample("phid", dist.Gamma(2.0, 2.0).expand([nb]))
    if y is None:
        return
    eta = a[bio] + u[yb] + s_phi * phi[cel] + s_th * th[cel] + jnp.sum(X * (beta + delta[bio]), axis=1)
    mu = jnp.exp(jnp.clip(eta, -20, 12)); zi = jax.nn.sigmoid(g0 + Z @ gamma)
    with numpyro.handlers.scale(scale=N / X.shape[0]):
        with numpyro.plate("obs", X.shape[0]):
            numpyro.sample("y", dist.ZeroInflatedDistribution(dist.NegativeBinomial2(mu, phid[bio]), gate=zi), obs=y)


def main():
    arq, steps, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    fold = int(sys.argv[4]) if len(sys.argv) > 4 else None
    pan = pd.read_parquet(arq); pan, _, _ = preparar(pan)
    y = pan.focos.values.astype(float); FX = CLIM + ["conv"] + LU
    X = pan[FX].values.astype(np.float32); Z = pan[["elev", "slope", "lroad", "lpop"]].values.astype(np.float32)
    bios = sorted(pan.bioma.unique()); bio = pan.bioma.map({b: i for i, b in enumerate(bios)}).values
    anos = sorted(pan.year.unique()); yb = pan.year.map({a: i for i, a in enumerate(anos)}).values * len(bios) + bio
    ix = np.floor(pan.x / 0.05 + 1e-6).astype(int).values; iy = np.floor(pan.y / 0.05 + 1e-6).astype(int).values
    key = pd.Series(list(zip(ix, iy))); cel, uniq = pd.factorize(key); ncel = len(uniq)
    pos = {k: i for i, k in enumerate(uniq)}
    E = []
    for (cx, cy), i in pos.items():
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            j = pos.get((cx + dx, cy + dy))
            if j is not None: E.append((i, j))
    E = np.array(E); E0, E1 = jnp.asarray(E[:, 0]), jnp.asarray(E[:, 1])
    nb, nyb, N = len(bios), int(yb.max()) + 1, len(y)
    train = np.ones(N, bool); te = None
    if fold is not None:
        tr_i, te_i = folds_spatial(pan.x.values, pan.y.values, 1.0, 5, seed=0)[fold]
        train[:] = False; train[tr_i] = True; te = te_i
    idx_tr = np.where(train)[0]; Ntr = len(idx_tr)
    cel_tr = np.zeros(ncel, bool); cel_tr[np.unique(cel[train])] = True
    Ef = E[cel_tr[E[:, 0]] & cel_tr[E[:, 1]]]          # ICAR ajustado só no grafo de treino (teste fora da soma zero)
    E0, E1 = jnp.asarray(Ef[:, 0]), jnp.asarray(Ef[:, 1]); MSK = jnp.asarray(cel_tr.astype(np.float32))
    print(f"n={N} treino={Ntr} células={ncel} arestas={len(E)} fold={fold}", flush=True)
    guide = AutoNormal(model, init_loc_fn=init_to_uniform(radius=0.0), init_scale=0.05)
    sched = lambda i: 0.01 * 0.05 ** (i / steps)
    svi = SVI(model, guide, numpyro.optim.ClippedAdam(sched), Trace_ELBO())
    rng = np.random.default_rng(0); B = 40000
    def args(ii):
        return (jnp.asarray(X[ii]), jnp.asarray(Z[ii]), jnp.asarray(bio[ii]), jnp.asarray(yb[ii]), jnp.asarray(cel[ii]), E0, E1, nb, nyb, ncel, Ntr, MSK)
    ii = idx_tr[rng.integers(0, Ntr, B)]
    st = svi.init(jax.random.PRNGKey(0), *args(ii), y=jnp.asarray(y[ii]))
    upd = jax.jit(lambda s, Xb, Zb, bb, ybb, cb, ob: svi.update(s, Xb, Zb, bb, ybb, cb, E0, E1, nb, nyb, ncel, Ntr, MSK, y=ob))
    t0 = time.time(); L = []
    for k in range(steps):
        ii = idx_tr[rng.integers(0, Ntr, B)]
        st, l = upd(st, jnp.asarray(X[ii]), jnp.asarray(Z[ii]), jnp.asarray(bio[ii]), jnp.asarray(yb[ii]), jnp.asarray(cel[ii]), jnp.asarray(y[ii])); L.append(float(l))
        if k % 2000 == 0: print(f"  passo {k}: ELBO {np.mean(L[-200:]):.0f} ({time.time()-t0:.0f}s)", flush=True)
    params = svi.get_params(st); post = guide.sample_posterior(jax.random.PRNGKey(1), params, sample_shape=(300,))
    post = {k: np.asarray(v) for k, v in post.items()}
    beta = post["beta"]; delta = post["delta"] * post["tau_d"][:, None, :]
    rows = []
    for j, f in enumerate(FX):
        for b, nm in enumerate(bios):
            e = beta[:, j] + delta[:, b, j]; rows.append({"variavel": f, "bioma": nm, "media": e.mean(), "q025": np.quantile(e, .025), "q975": np.quantile(e, .975)})
    pd.DataFrame(rows).to_csv(os.path.join(D, f"bym_efeitos_{tag}.csv"), index=False)
    json.dump({"s_phi": float(post["s_phi"].mean()), "s_th": float(post["s_th"].mean()), "s_u": float(post["s_u"].mean()), "elbo_final": float(np.mean(L[-500:]))},
              open(os.path.join(D, f"bym_var_{tag}.json"), "w"), indent=1)
    if te is not None:
        # células de teste: phi = média condicional do ICAR dados os vizinhos observados (interpolação harmônica);
        # theta = 0 com correção lognormal exp(s_th^2/2)
        import scipy.sparse as sp
        A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(ncel, ncel)).tocsr()
        deg = np.asarray(A.sum(1)).ravel(); test_cells = np.where(~cel_tr & (deg > 0))[0]; At = A[test_cells]
        mus = []
        for s in range(0, 300, 15):
            ph = post["phi"][s].copy(); ph[test_cells] = ph[cel_tr].mean()
            for _ in range(600):
                ph[test_cells] = (At @ ph) / deg[test_cells]
            eta = (post["a"][s][bio[te]] + post["u"][s][yb[te]] * post["s_u"][s] + post["s_phi"][s] * ph[cel[te]] + 0.5 * post["s_th"][s] ** 2
                   + np.sum(X[te] * (beta[s] + delta[s][bio[te]]), 1))
            zi = 1 / (1 + np.exp(-(post["g0"][s] + Z[te] @ post["gamma"][s])))
            mus.append(np.exp(np.clip(eta, -20, 12)) * (1 - zi))
        mu = np.mean(mus, 0)
        # diagnóstico: mesma fórmula nas linhas de treino (phi ajustado) e decomposição no teste
        s = 0; tri = idx_tr[::20]
        def comp(rows, ph):
            return dict(a=post["a"][s][bio[rows]], u=post["u"][s][yb[rows]] * post["s_u"][s], phi=post["s_phi"][s] * ph[cel[rows]],
                        th=post["s_th"][s] * post["th"][s][cel[rows]], xb=np.sum(X[rows] * (beta[s] + delta[s][bio[rows]]), 1))
        ct = comp(tri, post["phi"][s]); eta_tr = sum(ct.values())
        zi_tr = 1 / (1 + np.exp(-(post["g0"][s] + Z[tri] @ post["gamma"][s])))
        print("treino: soma obs", y[tri].sum(), "soma pred", float((np.exp(eta_tr) * (1 - zi_tr)).sum()), flush=True)
        ce = comp(te, ph)
        print("médias componentes treino:", {k: round(float(v.mean()), 3) for k, v in ct.items()}, flush=True)
        print("médias componentes teste :", {k: round(float(v.mean()), 3) for k, v in ce.items()}, flush=True)
        print("top 0,1% mu teste somam", float(np.sort(mu)[-len(mu)//1000:].sum()), "de", float(mu.sum()), flush=True)
        np.savez_compressed(os.path.join(D, f"bym_diag_fold{fold}.npz"), mu=mu, te=te, ph=ph, cel_tr=cel_tr)
        m = metrics(y[te], mu, y[train].mean()); m.update(fold=fold, modelo="ZINB espacial BYM")
        pd.DataFrame([m]).to_csv(os.path.join(D, f"bym_cv_fold{fold}.csv"), index=False); print(m, flush=True)
    print(pd.DataFrame(rows).pivot(index="variavel", columns="bioma", values="media").round(3).to_string(), flush=True)
    print(json.load(open(os.path.join(D, f"bym_var_{tag}.json"))), flush=True)


if __name__ == "__main__":
    main()
