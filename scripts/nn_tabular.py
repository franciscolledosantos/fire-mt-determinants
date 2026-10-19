"""Redes neurais tabulares no painel 0,05° × ano, mesma validação dos outros modelos.
- MLP (3 camadas, SiLU, dropout) e FT-Transformer (Gorishniy et al. 2021: tokenizador por variável + CLS + encoder transformer),
  ambos com perda de Poisson (saída = log da taxa).
- Esquemas: blocos espaciais de 1° (5 folds, mesmos da CV principal), temporal (treino 2010–19, teste 2020–24) e prospectivo 2025.
- Parada antecipada em 10% dos blocos de treino (validação interna espacial).
Uso: python nn_tabular.py <mlp|ft> [n_treino_max]"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch, torch.nn as nn
from cv_modelos import preparar, D
from cv_lib import folds_spatial, folds_temporal, metrics

torch.set_num_threads(2); torch.manual_seed(0)
ARQ = sys.argv[1]; NMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

class MLP(nn.Module):
    def __init__(s, p):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(p, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.1),
                            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(s, x): return s.f(x).squeeze(-1)

class FT(nn.Module):
    def __init__(s, p, d=32, L=2, h=4):
        super().__init__()
        s.w = nn.Parameter(torch.randn(p, d) * 0.1); s.b = nn.Parameter(torch.zeros(p, d)); s.cls = nn.Parameter(torch.zeros(1, 1, d))
        lay = nn.TransformerEncoderLayer(d, h, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(lay, L); s.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
    def forward(s, x):
        t = x.unsqueeze(-1) * s.w + s.b
        t = torch.cat([s.cls.expand(len(x), -1, -1), t], 1)
        return s.head(s.enc(t)[:, 0]).squeeze(-1)

def treinar(Xtr, ytr, xy_tr, kind, seed=0):
    rng = np.random.default_rng(seed)
    if len(ytr) > NMAX:
        k = rng.choice(len(ytr), NMAX, replace=False); Xtr, ytr, xy_tr = Xtr[k], ytr[k], xy_tr[k]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    blk = np.floor(xy_tr[:, 0]) * 1000 + np.floor(xy_tr[:, 1]); ub = np.unique(blk)
    val = np.isin(blk, rng.choice(ub, max(1, len(ub) // 10), replace=False))
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    m = MLP(Xt.shape[1]) if kind == "mlp" else FT(Xt.shape[1])
    with torch.no_grad():   # viés inicial = log da média
        last = m.f[-1] if kind == "mlp" else m.head[-1]; last.bias.fill_(float(np.log(ytr[~val].mean())))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3 if kind == "mlp" else 5e-4, weight_decay=1e-4)
    loss = nn.PoissonNLLLoss(log_input=True, full=False)
    tr_i = np.where(~val)[0]; va_i = np.where(val)[0]; best, best_state, bad = 1e18, None, 0
    bs = 2048 if kind == "mlp" else 1024; EP = 12 if kind == "mlp" else 8
    for ep in range(EP):
        m.train(); perm = rng.permutation(tr_i)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad(); l = loss(m(Xt[b]), yt[b]); l.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            lv = float(np.mean([loss(m(Xt[va_i[i:i + 8192]]), yt[va_i[i:i + 8192]]).item() for i in range(0, len(va_i), 8192)]))
        if lv < best - 1e-5: best, bad, best_state = lv, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 2: break
    m.load_state_dict(best_state); m.eval()
    def prever(X):
        with torch.no_grad():
            Z = torch.tensor((X - mu) / sd, dtype=torch.float32)
            return np.concatenate([torch.exp(m(Z[i:i + 8192])).numpy() for i in range(0, len(Z), 8192)])
    return prever, ep + 1

def main():
    kind = ARQ; t0 = time.time(); rows = []
    pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2024.parquet")); pan, _, Xb = preparar(pan)
    X = Xb.values.astype(np.float32); y = pan.focos.values.astype(np.float32); xy = pan[["x", "y"]].values
    esquemas = [("blocos espaciais 1° (5 folds)", folds_spatial(pan.x.values, pan.y.values, 1.0, 5)),
                ("temporal (teste a partir de 2020)", folds_temporal(pan.year.values, 2020))]
    for esq, folds in esquemas:
        for fi, (tr, te) in enumerate(folds):
            prever, ne = treinar(X[tr], y[tr], xy[tr], kind, seed=fi)
            m = metrics(y[te], prever(X[te]), y[tr].mean()); m.update(modelo=kind, esquema=esq, fold=fi, epocas=ne); rows.append(m)
            print(f"{esq} fold {fi}: D2 {m['D2']:.3f} ({ne} épocas, {time.time()-t0:.0f}s)", flush=True)
    p2 = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025.parquet")); p2, _, X2 = preparar(p2)
    tr = (p2.year <= 2024).values; te = (p2.year == 2025).values; X2 = X2.values.astype(np.float32); y2 = p2.focos.values.astype(np.float32)
    prever, ne = treinar(X2[tr], y2[tr], p2[["x", "y"]].values[tr], kind, seed=9)
    mu = prever(X2[te]); m = metrics(y2[te], mu, y2[tr].mean()); m.update(modelo=kind, esquema="prospectivo 2025", fold=0, epocas=ne, soma_pred=float(mu.sum())); rows.append(m)
    print(f"2025: D2 {m['D2']:.3f} soma pred {mu.sum():.0f} obs {y2[te].sum():.0f}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(D, f"nn_{kind}_cv.csv"), index=False)
    print(R.groupby("esquema")[["D2", "Spearman", "AUC_ocorr", "MAE"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
