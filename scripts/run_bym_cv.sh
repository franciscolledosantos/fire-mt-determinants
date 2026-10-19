cd "$(dirname "$0")/.."
for k in 0 1 2 3 4; do python3 -I scripts/bym_zinb.py resultados/painel_MT_2010_2024.parquet 8000 cv$k $k > resultados/bym_cv$k.log 2>&1; done
