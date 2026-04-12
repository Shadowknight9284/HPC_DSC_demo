# Hyperparameter Search HPC Demo

This folder contains a minimal, repo-ready demo for a 20–30 minute intro on hyperparameter search and HPC.

## Files
- `hyperparam_demo.py` — local or compute-node demo script comparing serial vs parallel `GridSearchCV`
- `slurm_run_demo.sh` — run the demo on one compute node via SLURM
- `slurm_array_demo.sh` — example SLURM array pattern where each task evaluates one hyperparameter configuration
- `train_one_config.py` — worker script used by the array job

## What the demo shows
1. Hyperparameter tuning means training many model configurations.
2. These configurations are largely independent, so the workload is easy to parallelize.
3. On one node, scikit-learn can parallelize with `n_jobs`.
4. On a cluster, SLURM job arrays can distribute configurations across tasks.

## Suggested live demo flow
1. Run `hyperparam_demo.py --n-jobs 1` to show the serial baseline.
2. Run `hyperparam_demo.py --n-jobs 8` to show parallel search on one node.
3. Show `slurm_array_demo.sh` and explain how each array index maps to one hyperparameter configuration.

## Local usage
```bash
python hyperparam_demo.py --n-jobs 1
python hyperparam_demo.py --n-jobs 8
```

## SLURM usage on Rutgers iLab
Single-node run:
```bash
sbatch slurm_run_demo.sh
```

Array-job example:
```bash
sbatch slurm_array_demo.sh
```

## Notes
- The demo uses the built-in breast cancer dataset from scikit-learn, so there is no data download step.
- The parameter grid is intentionally modest so the demo completes quickly and predictably.
- Adjust `--cpus-per-task` in the SLURM script to match the node allocation you get.
