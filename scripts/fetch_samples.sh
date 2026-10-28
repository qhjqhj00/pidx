#!/usr/bin/env bash
# 下载演示用的 PDF（来自 VectifyAI/PageIndex 仓库的 examples），并生成一份 3 页的"扫描件"。
# 用法：scripts/fetch_samples.sh   → 文件放在 samples/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p samples
BASE=https://raw.githubusercontent.com/VectifyAI/PageIndex/main/examples/documents
for f in 2023-annual-report.pdf q1-fy25-earnings.pdf earthmover.pdf; do
  if [ ! -s "samples/$f" ]; then
    echo "下载 $f"
    curl -fL --retry 3 -o "samples/$f" "$BASE/$f"
  fi
done
# 把 earthmover.pdf 前 3 页栅格化成纯图片 PDF，用来演示 OCR（MinerU）那条路径
PY=${PYTHON:-.venv/bin/python}
"$PY" - <<'PYEOF'
import pymupdf
src = pymupdf.open("samples/earthmover.pdf")
out = pymupdf.open()
for i in range(3):
    pix = src[i].get_pixmap(dpi=150)
    page = out.new_page(width=src[i].rect.width, height=src[i].rect.height)
    page.insert_image(page.rect, pixmap=pix)
out.save("samples/earthmover-scanned-3p.pdf")
print("生成 samples/earthmover-scanned-3p.pdf")
PYEOF
ls -la samples/*.pdf
