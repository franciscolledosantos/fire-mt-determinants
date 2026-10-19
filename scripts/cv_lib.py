"""Validação e modelos para o artigo de fogo em MT.

Funções reaproveitáveis: desenho de folds (aleatório, blocos espaciais, temporal),
métricas para contagens e os três modelos comparados (NB-GLM, ZINB, LightGBM).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- folds
def folds_random(n, k=5, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, k, n)
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def folds_spatial(x, y, block_deg=1.0, k=5, seed=0):
    """Blocos quadrados de block_deg graus atribuídos aleatoriamente a k folds.
    Todos os anos de um pixel caem no mesmo fold (o pixel nunca aparece em treino e teste)."""
    bx = np.floor(np.asarray(x) / block_deg).astype(int)
    by = np.floor(np.asarray(y) / block_deg).astype(int)
    blk = pd.Series(list(zip(bx, by))).astype(str).values
    ub = np.unique(blk)
    rng = np.random.default_rng(seed)
    assign = dict(zip(ub, rng.integers(0, k, len(ub))))
    f = np.array([assign[b] for b in blk])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def folds_temporal(years, test_from):
    years = np.asarray(years)
    return [(np.where(years < test_from)[0], np.where(years >= test_from)[0])]


def folds_leave_year_out(years):
    years = np.asarray(years)
    return [(np.where(years != y)[0], np.where(years == y)[0]) for y in np.unique(years)]


# ---------------------------------------------------------------- métricas
def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.sum(t - (y - mu))


def metrics(y, mu, ybar_train=None):
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    ref = np.full_like(y, ybar_train if ybar_train is not None else y.mean())
    d2 = 1 - poisson_dev(y, mu) / poisson_dev(y, ref)
    occ = (y > 0).astype(int)
    auc = roc_auc_score(occ, mu) if 0 < occ.mean() < 1 else np.nan
    rho = stats.spearmanr(y, mu).correlation
    return {"D2": d2, "AUC_ocorr": auc, "Spearman": rho,
            "MAE": np.mean(np.abs(y - mu)), "soma_obs": y.sum(), "soma_pred": mu.sum()}


def run_cv(fit_predict, X, y, folds, label):
    out = []
    for i, (tr, te) in enumerate(folds):
        mu = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
        m = metrics(y[te], mu, y[tr].mean()); m.update({"fold": i, "esquema": label, "n_teste": len(te)})
        out.append(m)
    return pd.DataFrame(out)
