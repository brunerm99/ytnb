#!/usr/bin/env bash
# Build the static site: every notebook is exported to a self-contained
# WASM (Pyodide) app under dist/<slug>/, meant to be iframed from
# marshallbruner.com. Data files in public/ are published alongside.
set -euo pipefail

cd "$(dirname "$0")"

MARIMO_VERSION="${MARIMO_VERSION:-0.23.14}"
OUT="${OUT:-dist}"

# One slug per notebook; each is published at /<slug>/
NOTEBOOKS=(
  gain
  reflectivity
)

rm -rf "$OUT"
mkdir -p "$OUT"

for slug in "${NOTEBOOKS[@]}"; do
  echo "==> exporting $slug.py"
  uvx "marimo@$MARIMO_VERSION" export html-wasm "$slug.py" \
    -o "$OUT/$slug" --mode run --show-code -f
done

# Data files, also served from a stable site-wide URL so any notebook (or
# anyone else) can fetch them: https://<site>/public/<file>
cp -R public "$OUT/public"
cp site/_headers site/_redirects "$OUT/"

echo "==> built $OUT"
