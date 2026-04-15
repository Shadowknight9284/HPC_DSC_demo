# gridsearch_multi.py
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    }

    gs = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,         # multi-core
        scoring="accuracy",
    )

    t0 = time.time()
    gs.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params = gs.best_params_
    best_cv_score = gs.best_score_
    test_acc = accuracy_score(y_test, gs.best_estimator_.predict(X_test))

    log_path = out_dir / "multi_core.log"
    with log_path.open("w") as f:
        f.write("mode=multi_core\n")
        f.write(f"n_configs={len(gs.cv_results_['mean_test_score'])}\n")
        f.write(f"elapsed_sec={elapsed:.3f}\n")
        f.write(f"best_cv_acc={best_cv_score:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
        f.write(f"best_params={best_params}\n")

    print(f"[multi_core] elapsed_sec={elapsed:.3f}, "
          f"best_cv_acc={best_cv_score:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
