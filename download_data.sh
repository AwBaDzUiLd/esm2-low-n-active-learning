#!/usr/bin/env bash
set -euo pipefail
mkdir -p data && cd data
BASE=https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3
[ -f DMS_substitutions.csv ] || wget -q $BASE/DMS_substitutions.csv
[ -d dms_assays ] || { wget -q $BASE/DMS_ProteinGym_substitutions.zip; unzip -q DMS_ProteinGym_substitutions.zip -d dms_assays; }
echo "data ready"