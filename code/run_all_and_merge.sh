#!/usr/bin/env bash
set -euo pipefail

echo "Running component solvers..."

python3 None.py
python3 Many.py
python3 Alternate.py
python3 some.py
python3 few.py

echo "All component CSVs generated."
echo "Merging results..."

python3 merge_results.py

echo "Done. Final table written to results.txt."
