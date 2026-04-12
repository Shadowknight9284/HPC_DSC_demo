#!/bin/bash
#SBATCH --job-name=hparam_array
#SBATCH --partition=unlimited
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --array=0-17
#SBATCH --output=array_%A_%a.out

set -euo pipefail

hostname
python --version

# If needed, activate your environment here.
python train_one_config.py
