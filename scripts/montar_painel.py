"""Monta o painel célula (0,05°) × ano para Mato Grosso, 2010–2024.

Fontes (todas públicas):
  focos    INPE BDQueimadas, satélite de referência, agregados por célula/mês (data/focos_INPE_...csv)
  uso      MapBiomas Coleção 10, contagem de pixels de 30 m por classe e célula (data/mapbiomas/*.bin)
  chuva    CHIRPS v2.0 diário 0,05° via ClimateSERV (chirps/raw/*.nc)
  estático relevo, distância a rodovias, densidade populacional e bioma da base do TCC (Pinheiro 2022)

Convenções: focos e clima no ano t; uso do solo no ano t-1 (antes do fogo do ano);
conversão recente = perda de vegetação nativa entre t-2 e t-1.
"""
import glob, os, pickle, sys
import numpy as np, pandas as pd

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(W, "resultados")
NX, NY, LON0, LAT0 = 228, 214, -61.65, -7.35
CODES = [3, 4, 5, 6, 9, 11, 12, 15, 20, 21, 24, 25, 29, 30, 33, 39, 40, 41, 46, 47, 48, 62, 23, 31, 32, 49, 50]
GRUPOS = {  # classe MapBiomas -> grupo do artigo
    "floresta": [3, 5, 6], "savana": [4], "campestre": [12], "alagado": [11],
    "pastagem": [15], "soja": [39], "outras_temp": [40, 41, 62], "cana": [20],
    "mosaico": [21], "perene_silvic": [9, 46, 47, 48], "urbano_outros": [23, 24, 25, 29, 30, 31, 32, 49, 50],
    "agua": [33],
}
NATIVOS = ["floresta", "savana", "campestre", "alagado"]


def cell_index(x, y):
    """índice (cx, cy) da célula a partir do centro (lon, lat)."""
    cx = np.floor((np.asarray(x) - LON0) / 0.05 + 1e-6).astype(int)
    cy = np.floor((LAT0 - np.asarray(y)) / 0.05 + 1e-6).astype(int)
    return cx, cy


def base_estatica():
    df = pickle.load(open(os.path.join(D, "samya_df.pkl"), "rb"))
    st = df.groupby(["x", "y"], as_index=False).agg(
        bioma=("bioma", "first"), elev=("elev", "first"), slope=("slope", "first"),
        roaddist=("roaddist", "first"), denspop=("denspop", "first"))
    st["bioma"] = st.bioma.astype(str)
    st["cx"], st["cy"] = cell_index(st.x, st.y)
    st["ix"] = np.floor(st.x / 0.05 + 1e-6).astype(int)
    st["iy"] = np.floor(st.y / 0.05 + 1e-6).astype(int)
    return st


def focos():
    f = pd.read_csv(os.path.join(D, "focos_INPE_satref_MT_grade005_2010_2025.csv"))
    a = f.groupby(["ix", "iy", "year"], as_index=False)[["n_all"]].sum().rename(columns={"n_all": "focos"})
    s = f[f.month.between(7, 10)].groupby(["ix", "iy", "year"], as_index=False)[["n_all"]].sum().rename(columns={"n_all": "focos_jul_out"})
    return a.merge(s, how="left").fillna({"focos_jul_out": 0})


def mapbiomas():
    out = []
    for fn in sorted(glob.glob(os.path.join(D, "mapbiomas", "*.bin"))):
        raw = open(fn, "rb").read()
        magic, year, nb, nc = np.frombuffer(raw[:16], dtype=np.int32)
        assert magic == 0x4C55 and nc == NX * NY
        a = np.frombuffer(raw[16:], dtype=np.uint16).reshape(nb, nc).astype(np.float64)
        valid = a[:-1].sum(0)  # exclui banda nodata (0)
        g = {}
        for k, codes in GRUPOS.items():
            g[k] = sum(a[CODES.index(c)] for c in codes if c in CODES)
        g["outros"] = a[len(CODES)]
        fr = pd.DataFrame({k: np.where(valid > 0, v / np.maximum(valid, 1), np.nan) for k, v in g.items()})
        fr["n_px"] = valid
        fr["cy"], fr["cx"] = np.divmod(np.arange(nc), NX)
        fr["year_lu"] = int(year)
        out.append(fr)
    return pd.concat(out, ignore_index=True) if out else None


def chirps():
    import xarray as xr
    cache = os.path.join(D, "chirps_indices.parquet")
    files = sorted(glob.glob(os.path.join(W, "chirps", "raw", "*.nc")))
    if os.path.exists(cache) and (not files or os.path.getmtime(cache) > max(os.path.getmtime(f) for f in files)):
        return pd.read_parquet(cache)
    rows = []
    for fn in files:
        da = xr.open_dataset(fn).precipitation_amount.load()
        y = int(da.time.dt.year.values[0]); lat = da.latitude.values; lon = da.longitude.values
        d = da.values  # (t, lat, lon)
        dry = d < 1.0
        run = np.zeros(d.shape[1:], int); cdd = np.zeros(d.shape[1:], int)
        for t in range(d.shape[0]):
            run = np.where(dry[t], run + 1, 0); cdd = np.maximum(cdd, run)
        mon = da.time.dt.month.values
        LO, LA = np.meshgrid(lon, lat)
        dfm = {"x": LO.ravel(), "y": LA.ravel(), "year": y, "cdd": cdd.ravel(), "dias_secos": dry.sum(0).ravel()}
        for m in range(1, 13):
            dfm[f"m{m:02d}"] = np.nansum(d[mon == m], 0).ravel()
        rows.append(pd.DataFrame(dfm))
        del da, d, dry
    c = pd.concat(rows, ignore_index=True)
    c["cx"], c["cy"] = cell_index(c.x, c.y)
    c = c.drop(columns=["x", "y"]); c.to_parquet(cache)
    return c


def montar(anos=range(2010, 2025)):
    st = base_estatica()
    pan = st.loc[st.index.repeat(len(anos))].reset_index(drop=True)
    pan["year"] = np.tile(list(anos), len(st))
    pan = pan.merge(focos(), on=["ix", "iy", "year"], how="left").fillna({"focos": 0, "focos_jul_out": 0})
    lu = mapbiomas()
    if lu is not None:
        l1 = lu.copy(); l1["year"] = l1.year_lu + 1
        pan = pan.merge(l1.drop(columns="year_lu"), on=["cx", "cy", "year"], how="left")
        nat = lu[["cx", "cy", "year_lu"]].copy(); nat["nativo"] = lu[NATIVOS].sum(1)
        n1 = nat.rename(columns={"nativo": "nat_t1"}); n1["year"] = n1.year_lu + 1
        n2 = nat.rename(columns={"nativo": "nat_t2"}); n2["year"] = n2.year_lu + 2
        pan = pan.merge(n1.drop(columns="year_lu"), on=["cx", "cy", "year"], how="left")
        pan = pan.merge(n2.drop(columns="year_lu"), on=["cx", "cy", "year"], how="left")
        pan["conversao"] = (pan.nat_t2 - pan.nat_t1).clip(lower=0)
    if glob.glob(os.path.join(W, "chirps", "raw", "*.nc")) or os.path.exists(os.path.join(D, "chirps_indices.parquet")):
        c = chirps()
        M = [f"m{m:02d}" for m in range(1, 13)]
        c["p_ann"] = c[M].sum(1)
        c["p_jjas"] = c[["m06", "m07", "m08", "m09"]].sum(1)
        c["p_ond"] = c[["m10", "m11", "m12"]].sum(1)
        c["p_jfma"] = c[["m01", "m02", "m03", "m04"]].sum(1)
        c["p_mjjas"] = c[["m05", "m06", "m07", "m08", "m09"]].sum(1)
        prev = c[["cx", "cy", "year", "p_ond"]].copy(); prev["year"] += 1
        c = c.merge(prev.rename(columns={"p_ond": "p_ond_ant"}), on=["cx", "cy", "year"], how="left")
        c["p_chuvosa_pre"] = c.p_ond_ant + c.p_jfma          # out(t-1)–abr(t): estação chuvosa que precede o fogo
        c["p_hidro"] = c.p_ond_ant + c.p_jfma + c.p_mjjas     # ano hidrológico out(t-1)–set(t)
        ref = c[c.year.between(2010, 2024)]
        clim = ref.groupby(["cx", "cy"])[["p_chuvosa_pre", "p_hidro", "cdd", "p_jjas"]].agg(["mean", "std"])
        clim.columns = ["_".join(k) for k in clim.columns]
        c = c.merge(clim.reset_index(), on=["cx", "cy"])
        for v in ["p_chuvosa_pre", "p_hidro", "cdd", "p_jjas"]:
            c[f"{v}_anom"] = (c[v] - c[f"{v}_mean"]) / c[f"{v}_std"]
        pan = pan.merge(c[["cx", "cy", "year", "cdd", "cdd_anom", "p_hidro", "p_hidro_anom", "p_chuvosa_pre", "p_chuvosa_pre_anom",
                           "p_jjas", "p_jjas_anom", "dias_secos"]], on=["cx", "cy", "year"], how="left")
    return pan


if __name__ == "__main__":
    pan = montar()
    pan.to_parquet(os.path.join(D, "painel_MT_2010_2024.parquet"))
    print(pan.shape); print(pan.isna().mean().round(3).to_string()); print(pan.groupby("year").focos.sum().to_string())
