#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Pack MilCOD7K for public release (native format only).
# Uses Python for zipping + checksums so it works on Windows Git Bash without
# needing the `zip` / `sha256sum` command-line tools.
#
# Usage:
#   bash tools/pack_release.sh <native_root> [out_dir]
#
#   native_root  native dir:  <root>/{train,val,test}/{Imgs,GT,Edge}
#   out_dir      where to write the archive (default: ./release)
#
# Produces:
#   MilCOD7K_native.zip   (Imgs/GT npy/Edge) + metadata.csv + SHA256SUMS.txt
# ---------------------------------------------------------------------------
set -e

NATIVE="${1:?usage: pack_release.sh <native_root> [out_dir]}"
OUT="${2:-./release}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$OUT"
cd "$OUT"

echo "==> metadata.csv"
python "$REPO_ROOT/tools/build_metadata.py" --data-root "$NATIVE" --out metadata.csv

echo "==> stage files"
rm -rf _pkg && mkdir -p _pkg
cp -r "$NATIVE" _pkg/MilCOD7K
# 只放纯数据（Imgs/GT/Edge），不往里塞 metadata/README/LICENSE
find _pkg -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "==> zip (via Python)"
python - <<'PY'
import os, zipfile
src = '_pkg'                       # pack the MilCOD7K/ folder inside _pkg
out = 'MilCOD7K_native.zip'
count = 0
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for root, dirs, files in os.walk(src):
        for fn in files:
            fp = os.path.join(root, fn)
            zf.write(fp, os.path.relpath(fp, src))   # archive paths: MilCOD7K/...
            count += 1
size = os.path.getsize(out)
print(f"  -> {out}: {count} files, {size/1024/1024:.1f} MB")
PY

echo "==> checksums (via Python)"
python - <<'PY'
import hashlib, os
files = ['MilCOD7K_native.zip', 'metadata.csv']
lines = []
for fn in files:
    if not os.path.exists(fn):
        continue
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    lines.append(f"{h.hexdigest()}  {fn}")
open('SHA256SUMS.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print("  -> SHA256SUMS.txt")
PY

rm -rf _pkg

echo ""
echo "Done. Files in: $OUT"
ls -lh
echo ""
echo "Next: upload MilCOD7K_native.zip + metadata.csv + SHA256SUMS.txt to your"
echo "      distribution host, then paste the share link + extraction code into"
echo "      README.md, docs/dataset.md, and checkpoints/README.md."
