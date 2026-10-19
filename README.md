# fire-mt-determinants

Code and result tables for the manuscript *Drought type, recent land conversion and fire in the Amazon, Cerrado and Pantanal of Mato Grosso, Brazil: implications for fire-risk management* (submitted to *Journal of Environmental Management*, 2026).

A public panel of 451,920 cell-years (0.05° cells, 2010–2024) combines INPE active fires, MapBiomas land use and CHIRPS rainfall for the three biomes of Mato Grosso. Hierarchical zero-inflated negative binomial models with spatial effects, geographically weighted models and LightGBM are validated by random folds, spatial blocks, a temporal test and the held-out year 2025. Dry-season rainfall deficits and native-vegetation conversion in the previous year are the main fire determinants, with biome-specific magnitudes; random validation overstates LightGBM skill relative to spatial blocks.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Panel construction, models, validation, sensitivity analyses, figures and the manuscript builder |
| `resultados/` | Result tables (CSV/JSON): cross-validation, effects, variances, GWNBR/GWPR local coefficients, SHAP, 2025 test, NB vs ZINB, revision files. Also the working folder for the panel parquet (not redistributed) |
| `figuras/` | Figs. 1–5 and S1–S2 of the manuscript; `fire_ecology_numbering/` holds the earlier numbering used by `rev17_figuras_jem.py` |

## Pipeline

1. `scripts/importar_chirps.py`, `importar_lu.py`, `importar_fogo.py` — import CHIRPS NetCDF, MapBiomas land-use and MapBiomas Fire rasters aggregated to the 0.05° grid.
2. `scripts/montar_painel.py` — builds `resultados/painel_MT_2010_2025.parquet` (cell × year) from INPE active fires, land use, rainfall indices and static covariates.
3. `scripts/cv_modelos.py` (+ `cv_lib.py`, `cv_glm_so.py`) — NB-GLM, ZINB and LightGBM under random, spatial-block and temporal validation → `cv_resultados_folds*.csv`, `cv_resumo*.csv`, `busca_hiperparametros.csv`.
4. `scripts/glmm_zinb.py`, `rodar_glmm.py`, `nuts_check.py`, `nuts_4cadeias.py` — hierarchical ZINB (NumPyro SVI/NUTS) → `glmm_*.csv/json`, `nuts_*.csv`. `bym_zinb.py` (`run_bym_cv.sh`) — BYM spatial effects → `bym_*.csv/json`.
5. `scripts/gwpr.py`, `gw_avancado.py` — geographically weighted Poisson / negative binomial → `gwpr_*.csv`, `gwnbr_*.csv`.
6. `scripts/nn_tabular.py`, `cnn_unet.py` (`run_nn.sh`) — MLP, FT-Transformer and U-Net comparison → `nn_*_cv.csv`. `shap_bioma.py` — SHAP by biome.
7. `scripts/sensibilidade_fogo.py`, `sensibilidade_fogo_ic.py`, `checagens_tecnicas.py`, `teste_2025.py`, `teste_2025_fogo_lag.py` — sensitivity, technical checks and the 2025 held-out test.
8. `scripts/rev17_zinb_vs_nb.py`, `rev17_teste2025_lags_anomalias.py`, `rev17_rotulo_pinheiro.py`, `rev17_figuras_jem.py` — revision analyses (NB vs ZINB, burned-area lags, Note S1) and the final figure set.
9. `scripts/figura_*.py`, `figuras_descritivas*.py` — figures; `scripts/build_jem_docx.py` — manuscript, supplement, highlights and cover letter (manuscript sources not included).

Run the scripts from any working directory: every path is resolved relative to the script's own location (`os.path.dirname(__file__)`).

## Data

Active fires: INPE BDQueimadas, reference satellite (https://terrabrasilis.dpi.inpe.br/queimadas/bdqueimadas/; open data, attribution). Land use and burned area: MapBiomas Collection 10 and MapBiomas Fire Collection 5 (https://brasil.mapbiomas.org; CC BY 4.0). Rainfall: CHIRPS v2.0 daily 0.05° via ClimateSERV (https://climateserv.servirglobal.net; public domain). Static covariates (relief, road distance, population density, biome) from the base of Pinheiro (2022). Place the raw files under `resultados/` as named in `scripts/montar_painel.py`; the panel parquets, NetCDF and binary rasters are git-ignored.

Raw input files are **not** redistributed in this repository; they are large and/or belong to third parties. Download them from the sources above and place them as described, or start from the derived tables in `resultados/`.

## Citation

See `CITATION.cff` (co-authors to be added). Licence: MIT.
