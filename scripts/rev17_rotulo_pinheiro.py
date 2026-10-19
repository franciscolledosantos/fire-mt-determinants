"""Revisão 17/10/2026 — reproduz a Nota S1: correlação entre focos do conjunto de Pinheiro (2022) no ano t
e focos INPE (satélite de referência, mesma célula de 0,05°) nos anos t e t−1."""
import os, json, numpy as np, pandas as pd
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resultados")
OUT = D
s = pd.read_pickle(os.path.join(D, "samya_df.pkl")); s.columns = [str(c) for c in s.columns]
s = s[["ix", "iy", "year", "focos"]]
f = pd.read_csv(os.path.join(D, "focos_INPE_satref_MT_grade005_2010_2025.csv"))
fa = f.groupby(["ix", "iy", "year"]).n_all.sum().rename("inpe").reset_index()
r = {}
for lag in [0, 1]:
    g = fa.copy(); g["year"] += lag
    m = s.merge(g, on=["ix", "iy", "year"], how="left").fillna({"inpe": 0})
    m = m[m.year - lag >= 2010]
    tot = m.groupby("year")[["focos", "inpe"]].sum()
    r[f"lag{lag}"] = {"anos": f"{int(m.year.min())}-{int(m.year.max())}", "n": int(len(m)),
                      "r_celula_ano": round(float(np.corrcoef(m.focos, m.inpe)[0, 1]), 3),
                      "r_totais_anuais": round(float(np.corrcoef(tot.focos, tot.inpe)[0, 1]), 3)}
print(json.dumps(r, indent=1)); json.dump(r, open(os.path.join(OUT, "rotulo_temporal_pinheiro.json"), "w"), indent=1)
