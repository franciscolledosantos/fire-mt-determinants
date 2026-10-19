"""ZINB hierárquico bayesiano (NumPyro) para focos anuais por célula.

log mu = a[bioma] + u[ano,bioma] + s[bloco 0,5°] + X (beta + delta[bioma])
logit pi (zero estrutural) = g0 + Z gamma
y ~ ZINB(pi, mu, phi[bioma])

Inferência variacional (SVI, AutoNormal) com minilotes sorteados fora do modelo e
verossimilhança escalada por N/B (estimador não viesado do ELBO).
"""
import time
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


USE_YEAR = True


def model(X, Z, bio, yb, blk, nb, nyb, nblk, N, y=None):
    p, q = X.shape[1], Z.shape[1]
    beta = numpyro.sample("beta", dist.Normal(0, 1).expand([p]))
    tau_d = numpyro.sample("tau_d", dist.HalfNormal(0.5).expand([p]))
    delta = numpyro.sample("delta", dist.Normal(0, 1).expand([nb, p])) * tau_d
    a = numpyro.sample("a", dist.Normal(0, 2).expand([nb]))
    s_u = numpyro.sample("s_u", dist.HalfNormal(1.0)); s_s = numpyro.sample("s_s", dist.HalfNormal(1.0))
    u = numpyro.sample("u", dist.Normal(0, 1).expand([nyb])) * s_u
    s = numpyro.sample("s", dist.Normal(0, 1).expand([nblk])) * s_s
    gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([q]))
    g0 = numpyro.sample("g0", dist.Normal(-1, 2))
    phi = numpyro.sample("phi", dist.Gamma(2.0, 2.0).expand([nb]))
    eta = a[bio] + (u[yb] if USE_YEAR else 0.0) + s[blk] + jnp.sum(X * (beta + delta[bio]), axis=1)
    mu = jnp.exp(jnp.clip(eta, -20, 12))
    zi = jax.nn.sigmoid(g0 + Z @ gamma)
    with numpyro.handlers.scale(scale=N / X.shape[0]):
        with numpyro.plate("obs", X.shape[0]):
            numpyro.sample("y", dist.ZeroInflatedDistribution(dist.NegativeBinomial2(mu, phi[bio]), gate=zi), obs=y)


def ajustar(X, Z, bio, yb, blk, y, steps=8000, lr=0.01, batch=40000, seed=0):
    nb, nyb, nblk, N = int(bio.max()) + 1, int(yb.max()) + 1, int(blk.max()) + 1, len(y)
    guide = AutoNormal(model)
    sched = lambda i: lr * 0.05 ** (i / steps)
    svi = SVI(model, guide, numpyro.optim.ClippedAdam(sched), Trace_ELBO())
    rng = np.random.default_rng(seed)
    def batch_args(ix):
        return (jnp.asarray(X[ix]), jnp.asarray(Z[ix]), jnp.asarray(bio[ix]), jnp.asarray(yb[ix]), jnp.asarray(blk[ix]))
    ix = rng.integers(0, N, batch)
    st = svi.init(jax.random.PRNGKey(seed), *batch_args(ix), nb, nyb, nblk, N, y=jnp.asarray(y[ix]))
    upd = jax.jit(lambda s, Xb, Zb, bb, yyb, kb, ob: svi.update(s, Xb, Zb, bb, yyb, kb, nb, nyb, nblk, N, y=ob))
    losses = []; t0 = time.time()
    for k in range(steps):
        ix = rng.integers(0, N, batch)
        st, l = upd(st, *batch_args(ix), jnp.asarray(y[ix])); losses.append(float(l))
        if k % 2000 == 0: print(f"  passo {k}: ELBO {np.mean(losses[-200:]):.0f} ({time.time()-t0:.0f}s)", flush=True)
    params = svi.get_params(st)
    post = guide.sample_posterior(jax.random.PRNGKey(1), params, sample_shape=(1000,))
    class R: pass
    r = R(); r.losses = np.array(losses); r.params = params
    print(f"SVI {steps} passos em {time.time()-t0:.0f}s; ELBO final {np.mean(losses[-500:]):.0f}", flush=True)
    return post, r
