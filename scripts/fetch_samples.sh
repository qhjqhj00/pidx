#!/usr/bin/env bash
# 下载演示用的 PDF（来自 VectifyAI/PageIndex 仓库的 examples），并生成一份 3 页的"扫描件"；
# 再下载几份老格式 Office 文件（doc/ppt/xls，Apache Tika 的测试样例），并生成多格式资料包（scripts/make_samples.py）。
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
    page.insert_image(page.rect, stream=pix.tobytes("jpeg", jpg_quality=80))
out.save("samples/earthmover-scanned-3p.pdf", deflate=True)
print("生成 samples/earthmover-scanned-3p.pdf")
PYEOF
mkdir -p samples/legacy
TIKA=https://raw.githubusercontent.com/apache/tika/main/tika-parsers/tika-parsers-standard/tika-parsers-standard-modules/tika-parser-microsoft-module/src/test/resources/test-documents
for f in testWORD.doc testPPT.ppt testEXCEL.xls; do
  [ -s "samples/legacy/$f" ] || curl -fL --retry 3 -o "samples/legacy/$f" "$TIKA/$f"
done
if "$PY" -c "import docx, pptx, openpyxl" 2>/dev/null; then
  "$PY" scripts/make_samples.py
else
  echo "跳过多格式资料包：先 .venv/bin/pip install -r requirements-dev.txt，再跑 scripts/make_samples.py"
fi
ls -la samples/*.pdf samples/legacy
