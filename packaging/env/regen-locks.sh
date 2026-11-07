#!/bin/sh
# Regenera packaging/env/locks/*.txt (versiones y hashes) para Linux y Windows con Python 3.12.
# Requiere `uv`. Una resolución común por plataforma + un lock por componente con esas mismas versiones.
set -e
cd "$(dirname "$0")"
mkdir -p locks
tmp="$(mktemp -d)"
for plat in linux windows; do
  files="base.in tui.in keras.in torch.in"
  [ "$plat" = linux ] && files="$files tflm.in"   # tflite-micro: solo Linux x86-64
  cat $files > "$tmp/all-$plat.in"
  uv pip compile "$tmp/all-$plat.in" --python-version 3.12 --python-platform "$plat" \
    --generate-hashes -q -o "$tmp/all-$plat.txt"
  for c in base tui keras torch tflm; do
    [ "$plat" = windows ] && [ "$c" = tflm ] && continue
    uv pip compile "$c.in" -c "$tmp/all-$plat.txt" --python-version 3.12 --python-platform "$plat" \
      --generate-hashes -q -o "locks/$c-$plat.txt"
  done
done
# En Windows PyPI solo tiene PyTorch para CPU: fijar la rueda CUDA del índice de PyTorch (con su hash)
python pin_torch_cuda.py --cuda cu130 --lock locks/torch-windows.txt
rm -rf "$tmp"
echo "locks regenerados"
