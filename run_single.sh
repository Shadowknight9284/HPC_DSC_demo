#!/bin/bash
#SBATCH --job-name=grid_single
#SBATCH --partition=unlimited
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=out/%j/slurm_%j.out

set -euo pipefail

cd "$HOME/HPC_DSC_demo/HPC_DSC_demo"

source "$HOME/hpc_demo_env/bin/activate"

mkdir -p out/"$SLURM_JOB_ID"

python gridsearch_single.py
