#!/bin/bash
#SBATCH --job-name=hparam_demo
#SBATCH --partition=unlimited
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=hparam_demo_%j.out

set -euo pipefail

hostname
python --version

# If you use conda or a module system on iLab, activate it here.
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv

python hyperparam_demo.py --n-jobs 1 --out results_serial.json
python hyperparam_demo.py --n-jobs ${SLURM_CPUS_PER_TASK} --out results_parallel.json

python - <<'PY'
import json
from pathlib import Path

serial = json.loads(Path('results_serial.json').read_text())
parallel = json.loads(Path('results_parallel.json').read_text())
speedup = serial['elapsed_seconds'] / parallel['elapsed_seconds']
print('\n=== SUMMARY ===')
print(f"Serial time:   {serial['elapsed_seconds']:.2f}s")
print(f"Parallel time: {parallel['elapsed_seconds']:.2f}s")
print(f"Speedup:       {speedup:.2f}x")
print(f"Best params:   {parallel['best_params']}")
print(f"Test accuracy: {parallel['test_accuracy']:.4f}")
PY
