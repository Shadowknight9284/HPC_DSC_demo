# Parallel Hyperparameter Search on HPC (Rutgers iLab Demo)

This mini‑repo shows how **hyperparameter tuning** can be sped up using **multiple CPU cores** on an HPC cluster.

We compare:

- a **single‑core** grid search (`n_jobs=1`), and
- a **multi‑core** grid search (`n_jobs=-1`)

on the **same dataset, model, and hyperparameter grid**, run through **SLURM** on Rutgers iLab. At the end we compute the speedup and check that model quality is unchanged.

---

## 1. High‑level idea

**Goal:**  
Show that hyperparameter search is:

- computationally expensive (many fits × k‑fold CV), and  
- naturally parallelizable across cores or nodes.

**Key ingredients:**

- **ML side**
  - `sklearn.datasets.load_breast_cancer`
  - `RandomForestClassifier`
  - `GridSearchCV` with a 36‑point hyperparameter grid and 5‑fold CV

- **HPC side**
  - SLURM job scripts
  - One job requesting **1 CPU**
  - One job requesting **8 CPUs** (or more) for the same code

We then compare the elapsed times and compute:

\[
\text{speedup} = \frac{T_{\text{single}}}{T_{\text{multi}}}
\]

---

## 2. Repository layout

```text
HPC_DSC_demo/
├── gridsearch_single.py   # single-core GridSearchCV (n_jobs=1)
├── gridsearch_multi.py    # multi-core GridSearchCV (n_jobs=-1)
├── compare_runs.py        # reads logs and computes speedup
├── run_single.sh          # SLURM script: 1 CPU, runs gridsearch_single.py
├── run_multi.sh           # SLURM script: 8 CPUs, runs gridsearch_multi.py
└── out/                   # logs and SLURM outputs (created at runtime)
```

---

## 3. Dataset, model, and hyperparameter grid

Both Python scripts share the exact same ML setup:

- **Dataset:** scikit‑learn **breast cancer** dataset (`load_breast_cancer`)
- **Split:** 80% train / 20% test, stratified, `random_state=42`
- **Model:** `RandomForestClassifier(random_state=42)`
- **Metric:** accuracy on the held‑out test set
- **Hyperparameter grid:**

  ```python
  param_grid = {
      "n_estimators":,[1]
      "max_depth": [None, 5, 10],
      "min_samples_split": ,
      "max_features": ["sqrt", "log2"],
  }
  ```

- **Grid size:**  
  `3 × 3 × 2 × 2 = 36` configurations

- **Cross‑validation:**  
  Each configuration is evaluated with **5‑fold CV**, so:

  **Total fits = 36 configs × 5 folds = 180 model fits**

This is what makes the computation heavy enough that parallelism noticeably helps.

---

## 4. `gridsearch_single.py` — single‑core baseline

**Purpose:** Run `GridSearchCV` on **one core** as a baseline.

What it does:

1. Creates `out/` if it doesn’t exist.
2. Loads the breast cancer dataset and does a stratified 80/20 split.
3. Instantiates a `RandomForestClassifier(random_state=42)`.
4. Defines the **same** 36‑point hyperparameter grid.
5. Sets up `GridSearchCV` with:

   ```python
   gs = GridSearchCV(
       estimator=rf,
       param_grid=param_grid,
       cv=5,
       n_jobs=1,          # single-core
       scoring="accuracy",
   )
   ```

6. Times `gs.fit(X_train, y_train)` using `time.time()` (or `time.perf_counter()`).
7. After fitting, it extracts:
   - `best_params_` (best hyperparameters)
   - `best_score_` (best CV accuracy)
   - test accuracy on `X_test`
8. Writes a log file: `out/single_core.log` with:

   ```text
   mode=single_core
   n_configs=36
   elapsed_sec=...
   best_cv_acc=...
   test_acc=...
   best_params={...}
   ```

9. Prints a short summary to stdout so you can see the result immediately.

This script is the **baseline**: one core, no parallelism.

---

## 5. `gridsearch_multi.py` — multi‑core run

**Purpose:** Run the exact same `GridSearchCV`, but using **all available CPUs** on the node.

Differences vs `gridsearch_single.py`:

- Uses the same:
  - dataset and split,
  - model,
  - hyperparameter grid,
  - CV folds,
  - logging format.

- The only intentional code change is:

  ```python
  gs = GridSearchCV(
      estimator=rf,
      param_grid=param_grid,
      cv=5,
      n_jobs=-1,         # multi-core: use all allocated cores
      scoring="accuracy",
  )
  ```

- Times `gs.fit(...)` in the same way as the single-core script.
- Writes a log file: `out/multi_core.log` with the **same fields**:

   ```text
   mode=multi_core
   n_configs=36
   elapsed_sec=...
   best_cv_acc=...
   test_acc=...
   best_params={...}
   ```

Because everything except `n_jobs` is identical, any change in **wall‑clock time** reflects parallelism, not a change in the ML setup.

---

## 6. SLURM scripts — running on Rutgers iLab

We have two SLURM scripts that submit these Python scripts to the cluster.

### 6.1 `run_single.sh` — 1 CPU

```bash
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

mkdir -p "out/$SLURM_JOB_ID"
cd "out/$SLURM_JOB_ID"

python ../../gridsearch_single.py
```

**Key points:**

- `--cpus-per-task=1` → SLURM gives this job **1 CPU**.
- `--ntasks=1` → it’s a single process (one Python process).
- We activate the Python env, ensure `out/$SLURM_JOB_ID` exists, cd into it, and run the single‑core script.
- Log files and SLURM output for this job all live under the job‑specific directory.

Submit with:

```bash
sbatch run_single.sh
```

---

### 6.2 `run_multi.sh` — 8 CPUs

```bash
#!/bin/bash
#SBATCH --job-name=grid_multi
#SBATCH --partition=unlimited
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=out/%j/slurm_%j.out

set -euo pipefail

cd "$HOME/HPC_DSC_demo/HPC_DSC_demo"
source "$HOME/hpc_demo_env/bin/activate"

mkdir -p "out/$SLURM_JOB_ID"
cd "out/$SLURM_JOB_ID"

python ../../gridsearch_multi.py
```

**Key points:**

- `--cpus-per-task=8` → SLURM gives this job **8 CPUs**.
- `GridSearchCV(n_jobs=-1)` uses all CPUs the job can see, so it will parallelize across those 8 cores.
- Memory is bumped to `8G` to safely handle more workers.

Submit with:

```bash
sbatch run_multi.sh
```

Monitor jobs with:

```bash
squeue -u $USER
```

---

## 7. `compare_runs.py` — measuring speedup

**Purpose:** Read the logs from the two runs and compute the speedup.

What it does:

1. Reads `out/single_core.log` and `out/multi_core.log`.
2. Parses key fields:

   - `n_configs`
   - `elapsed_sec`
   - `best_cv_acc`
   - `test_acc`

3. Asserts both runs used the same number of configs (should be 36).
4. Computes:

   ```python
   speedup = elapsed_single / elapsed_multi
   ```

5. Prints a summary, for example:

   ```text
   Hyperparameter search comparison (same grid, same data)
   Configs evaluated: 36

   Single-core:  elapsed = 31.438 s
   Multi-core:   elapsed =  6.500 s
   Speedup (single / multi) = 4.83x

   Single-core best CV acc: 0.9604
   Multi-core  best CV acc: 0.9604
   Single-core test acc:    0.9561
   Multi-core  test acc:    0.9561
   ```

This gives you:

- One clean **speedup number**, and
- A check that **accuracy is essentially the same** in both runs.

Run it with:

```bash
python compare_runs.py
```

(from the project root; it expects logs under `out/`).

---

## 8. How to run the full demo

From the HPC login node:

1. `cd` into the project directory:

   ```bash
   cd "$HOME/HPC_DSC_demo/HPC_DSC_demo"
   ```

2. Submit the single‑core job:

   ```bash
   sbatch run_single.sh
   ```

3. Submit the multi‑core job:

   ```bash
   sbatch run_multi.sh
   ```

4. Watch the jobs:

   ```bash
   squeue -u $USER
   ```

5. Once both jobs are finished, inspect `out/` to see the per‑job directories and logs.

6. Back on your laptop or on the head node, run:

   ```bash
   python compare_runs.py
   ```

7. Take the printed numbers (times, speedup, accuracies) and drop them into the “Results” slide.

---

## 9. What to emphasize when presenting

- **Brute‑force tuning:** Grid search evaluates every combination in the parameter grid with cross‑validation.
- **Cost:** For this demo, 36 configs × 5 folds = 180 model fits.
- **Parallelism:** Every configuration and CV fold is largely independent → easy to parallelize.
- **Code:** Single vs multi‑core differ in only one argument: `n_jobs=1` vs `n_jobs=-1`.
- **HPC:** On SLURM, you control how many cores you get with `--cpus-per-task`; scikit‑learn uses them when `n_jobs=-1` is set.
- **Result:** Multi‑core run is several times faster, with essentially the **same best model and accuracy**.

That’s the full story this repo is meant to demonstrate.