cd "$(dirname "$0")/.."
while pgrep -f "nn_tabular.py mlp" > /dev/null; do sleep 10; done
python3 -I scripts/nn_tabular.py ft 200000 > resultados/nn_ft.log 2>&1
python3 -I scripts/cnn_unet.py > resultados/nn_unet.log 2>&1
