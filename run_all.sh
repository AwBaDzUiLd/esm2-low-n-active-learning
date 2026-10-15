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
python active_hybrid.py | tee active_hybrid_out.txt
python build_pools.py | tee build_pools_out.txt
python step12_transfer.py | tee step12_transfer_out.txt
python step12_active.py | tee step12_active_out.txt
python holdout_prepare.py | tee holdout_prepare_out.txt
python holdout_active.py | tee holdout_active_out.txt
python make_report.py