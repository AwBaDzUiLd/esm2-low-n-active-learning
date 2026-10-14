#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash download_data.sh
python load_assays.py
python baselines.py
python embed.py
python lowN.py
python analysis.py | tee analysis_out.txt
python active.py | tee active_out.txt
python make_report.py