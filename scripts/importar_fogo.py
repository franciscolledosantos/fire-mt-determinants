"""Importa a área queimada anual do MapBiomas Fogo Coleção 5 (contagem de pixels de 30 m queimados por célula de 0,05°,
gerada no navegador a partir de burned_area_AAAA.tif) e junta ao painel.
Arquivo: cabeçalho int32 [0x4642, ano0, nanos, ncel] + uint32 [nanos × ncel], célula = cy*NX + cx (mesma grade de montar_painel).
Saídas: data/fogo_c5_celula_ano.parquet e data/painel_MT_2010_2025_fogo.parquet."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from montar_painel import NX, NY, LON0, LAT0, cell_index

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "resultados")
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "mapbiomas_fogo_c5_MT005_2009_2025.bin")
raw = open(fn, "rb").read(); magic, y0, ny, nc = np.frombuffer(raw[:16], np.int32)
assert magic == 0x4642 and nc == NX * NY
cnt = np.frombuffer(raw[16:], np.uint32).reshape(ny, nc).astype(np.float64)
# pixels por célula (centro do pixel dentro da célula), grade MapBiomas: origem e resolução do GeoTIFF
OX, OY, R = -74.54381924374933, 6.777788818681791, 0.0002694945852358564
def n_centros(a, b, o, r, sinal):
    # número de inteiros i com o + sinal*(i+0.5)*r em [a, b)
    if sinal > 0:
        lo = np.ceil((a - o) / r - 0.5); hi = np.ceil((b - o) / r - 0.5)
    else:
        lo = np.ceil((o - b) / r - 0.5); hi = np.ceil((o - a) / r - 0.5)
    return hi - lo
nx_px = np.array([n_centros(LON0 + i * 0.05, LON0 + (i + 1) * 0.05, OX, R, 1) for i in range(NX)])
ny_px = np.array([n_centros(LAT0 - (j + 1) * 0.05, LAT0 - j * 0.05, OY, R, -1) for j in range(NY)])
tot = (ny_px[:, None] * nx_px[None, :]).ravel()
rows = []
for k in range(ny):
    cy, cx = np.divmod(np.arange(nc), NX)
    rows.append(pd.DataFrame({"cx": cx, "cy": cy, "year": y0 + k, "queim_px": cnt[k], "queim_frac": cnt[k] / tot}))
F = pd.concat(rows, ignore_index=True); F.to_parquet(os.path.join(D, "fogo_c5_celula_ano.parquet"))
print("fração máx:", F.queim_frac.max().round(4), " pixels/célula:", tot.min(), tot.max())

pan = pd.read_parquet(os.path.join(D, "painel_MT_2010_2025.parquet"))
pan = pan.merge(F[["cx", "cy", "year", "queim_frac"]], on=["cx", "cy", "year"], how="left")
for L in (1, 2):
    f = F[["cx", "cy", "year", "queim_frac"]].copy(); f["year"] += L
    pan = pan.merge(f.rename(columns={"queim_frac": f"queim_t{L}"}), on=["cx", "cy", "year"], how="left")
pan["area_km2"] = (0.05 * 111.32) ** 2 * np.cos(np.deg2rad(pan.y))
pan["queim_ha"] = pan.queim_frac * pan.area_km2 * 100
pan.to_parquet(os.path.join(D, "painel_MT_2010_2025_fogo.parquet"))
print(pan.groupby(["year", "bioma"]).queim_ha.sum().unstack().div(1e3).round(0).to_string())
