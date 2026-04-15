# compare_runs.py
from pathlib import Path

def load_log(path: Path):
    data = {}
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k] = v
    return data

def main():
    base = Path("out")
    single_log = load_log(base / "single_core.log")
    multi_log  = load_log(base / "multi_core.log")

    if not single_log or not multi_log:
        print("Run both gridsearch_single.py and gridsearch_multi.py first.")
        return

    n_configs = int(single_log["n_configs"])
    assert n_configs == int(multi_log["n_configs"]), "Mismatch in number of configs"

    t_single = float(single_log["elapsed_sec"])
    t_multi  = float(multi_log["elapsed_sec"])

    speedup = t_single / t_multi if t_multi > 0 else float("inf")

    print("Hyperparameter search comparison (same grid, same data)")
    print(f"Configs evaluated: {n_configs}")
    print()
    print(f"Single-core:  elapsed = {t_single:.3f} s")
    print(f"Multi-core:   elapsed = {t_multi:.3f} s")
    print(f"Speedup (single / multi) = {speedup:.2f}x")
    print()
    print(f"Single-core best CV acc: {single_log['best_cv_acc']}")
    print(f"Multi-core  best CV acc: {multi_log['best_cv_acc']}")
    print(f"Single-core test acc:    {single_log['test_acc']}")
    print(f"Multi-core  test acc:    {multi_log['test_acc']}")


if __name__ == "__main__":
    main()
