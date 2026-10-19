"""Rede convolucional (U-Net pequena) sobre a grade de 0,05° de Mato Grosso: cada ano é um mapa de 214 × 228 células com
22 camadas de covariáveis (uso do solo t−1, conversão, anomalias de chuva, relevo, acesso, população, bioma) e a saída é o
log da taxa de focos em cada célula (perda de Poisson só nas células de treino). A rede vê a vizinhança das covariáveis
(campo receptivo de ~2°), nunca os focos. Mesma validação: blocos de 1° (5 folds), temporal 2020–24 e prospectivo 2025."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from cv_modelos import preparar, D
from cv_lib import folds_spatial, folds_temporal, metrics
from montar_painel import NX, NY

torch.set_num_threads(2); torch.manual_seed(0)
H, W = 216, 232   # grade 214 × 228 com borda para múltiplos de 8

def grade(pan, Xb):
    anos = sorted(pan.year.unique()); ia = {a: i for i, a in enumerate(anos)}
    C = Xb.shape[1]; G = np.zeros((len(anos), C + 1, H, W), np.float32); Y = np.zeros((len(anos), H, W), np.float32)
    M = np.zeros((len(anos), H, W), bool)
    t = pan.year.map(ia).values; cy, cx = pan.cy.values, pan.cx.values
    G[t, :C, cy, cx] = Xb.values.astype(np.float32); G[t, C, cy, cx] = 1.0; Y[t, cy, cx] = pan.focos.values; M[t, cy, cx] = True
    mu = G[:, :C][M[:, None].repeat(C, 1)].reshape(-1) if False else None
    return anos, G, Y, M, t, cy, cx

def bloco(c, k, d=0.0):
    return nn.Sequential(nn.Conv2d(c, k, 3, padding=1), nn.GroupNorm(8, k), nn.SiLU(), nn.Dropout2d(d), nn.Conv2d(k, k, 3, padding=1), nn.GroupNorm(8, k), nn.SiLU())

class UNet(nn.Module):
    def __init__(s, cin, b=16):
        super().__init__()
        s.e1, s.e2, s.e3 = bloco(cin, b), bloco(b, 2 * b), bloco(2 * b, 4 * b, 0.1)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), bloco(4 * b, 2 * b)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), bloco(2 * b, b)
        s.out = nn.Conv2d(b, 1, 1)
    def forward(s, x):
        a = s.e1(x); b_ = s.e2(F.max_pool2d(a, 2)); c = s.e3(F.max_pool2d(b_, 2))
        d = s.d2(torch.cat([s.u2(c), b_], 1)); e = s.d1(torch.cat([s.u1(d), a], 1))
        return s.out(e).squeeze(1)

def treinar(G, Y, Mtr, Mva, anos_tr, seed=0, passos=600, crop=112):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Gt, Yt = torch.tensor(G), torch.tensor(Y); Mt, Mv = torch.tensor(Mtr), torch.tensor(Mva)
    m = UNet(G.shape[1]);
    with torch.no_grad(): m.out.bias.fill_(float(np.log(Y[Mtr].mean())))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=passos)
    def perda(idx, mask, r0=None, c0=None):
        if r0 is None: g, yy, mk = Gt[idx], Yt[idx], mask[idx]
        else:
            g, yy, mk = Gt[idx, :, r0:r0 + crop, c0:c0 + crop], Yt[idx, r0:r0 + crop, c0:c0 + crop], mask[idx, r0:r0 + crop, c0:c0 + crop]
        if mk.sum() == 0: return None
        eta = m(g); mu = torch.exp(eta.clamp(-15, 10))
        return ((mu - yy * eta)[mk]).mean()
    best, best_state = 1e18, None
    for k in range(passos):
        m.train(); idx = torch.tensor(rng.choice(anos_tr, 4, replace=False))
        r0, c0 = 4 * rng.integers(0, (H - crop) // 4 + 1), 4 * rng.integers(0, (W - crop) // 4 + 1)
        opt.zero_grad(); l = perda(idx, Mt, r0, c0)
        if l is not None: l.backward(); opt.step()
        sch.step()
        if (k + 1) % 50 == 0:
            m.eval()
            with torch.no_grad(): lv = float(perda(torch.tensor(anos_tr), Mv))
            if lv < best: best, best_state = lv, {a: v.clone() for a, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    def prever(ia):
        with torch.no_grad(): return torch.exp(m(Gt[torch.tensor(ia)]).clamp(-15, 10)).numpy()
    return prever

def avaliar(prever, ia_te, mask_te, Y, ybar):
    P = prever(ia_te); y = Y[ia_te][mask_te[ia_te]]; mu = P[mask_te[ia_te]]
    return metrics(y, mu, ybar), mu

def main():
    t0 = time.time(); rows = []
    pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025.parquet")); pan, _, Xb = preparar(pan)
    anos, G, Y, M, t, cy, cx = grade(pan, Xb); A = np.array(anos)
    rng = np.random.default_rng(0)
    ok = pan.year.values <= 2024; p24 = pan[ok]
    # 1) blocos de 1° (mesmos folds da CV principal, construídos sobre o painel 2010–2024)
    for fi, (tr, te) in enumerate(folds_spatial(p24.x.values, p24.y.values, 1.0, 5)):
        Mte = np.zeros_like(M); Mte[t[ok][te], cy[ok][te], cx[ok][te]] = True
        Mtr_all = M & ~Mte; Mtr_all[A == 2025] = False
        cel_tr = np.unique(cy[ok][tr] * 1000 + cx[ok][tr])
        vb = np.isin((np.floor(p24.x.values[tr]) * 1000 + np.floor(p24.y.values[tr])),
                     rng.choice(np.unique(np.floor(p24.x.values[tr]) * 1000 + np.floor(p24.y.values[tr])), 10, replace=False))
        Mva = np.zeros_like(M); Mva[t[ok][tr][vb], cy[ok][tr][vb], cx[ok][tr][vb]] = True
        prever = treinar(G, Y, Mtr_all & ~Mva, Mva, np.where(A <= 2024)[0], seed=fi)
        m, _ = avaliar(prever, np.where(A <= 2024)[0], Mte, Y, Y[Mtr_all].mean()); m.update(modelo="U-Net", esquema="blocos espaciais 1° (5 folds)", fold=fi)
        rows.append(m); print(f"bloco fold {fi}: D2 {m['D2']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    # 2) temporal e 3) 2025: validação interna = 10% das células
    cel = np.unique(cy * 1000 + cx); vc = set(rng.choice(cel, len(cel) // 10, replace=False))
    Mv_cel = np.zeros((H, W), bool)
    for c in vc: Mv_cel[c // 1000, c % 1000] = True
    for esq, ano_tr, ano_te in [("temporal (teste a partir de 2020)", A <= 2019, (A >= 2020) & (A <= 2024)), ("prospectivo 2025", A <= 2024, A == 2025)]:
        Mtr = M & ano_tr[:, None, None] & ~Mv_cel[None]; Mva = M & ano_tr[:, None, None] & Mv_cel[None]
        prever = treinar(G, Y, Mtr, Mva, np.where(ano_tr)[0], seed=7)
        Mte = M & ano_te[:, None, None]
        m, mu = avaliar(prever, np.where(ano_te)[0], Mte, Y, Y[M & ano_tr[:, None, None]].mean())
        m.update(modelo="U-Net", esquema=esq, fold=0); rows.append(m); print(f"{esq}: D2 {m['D2']:.3f} soma pred {mu.sum():.0f} ({time.time()-t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "nn_unet_cv.csv"), index=False)
    print(R.groupby("esquema")[["D2", "Spearman", "AUC_ocorr", "MAE"]].mean().round(3).to_string())

if __name__ == "__main__":
    main()
