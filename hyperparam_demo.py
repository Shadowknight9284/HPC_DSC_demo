import argparse
import json
import time
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for GridSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--out", type=str, default="results_demo.json", help="Output JSON path")
    args = parser.parse_args()

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }

    model = RandomForestClassifier(random_state=42)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=args.cv,
        n_jobs=args.n_jobs,
        verbose=1,
        return_train_score=False,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = search.best_estimator_
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    result = {
        "n_jobs": args.n_jobs,
        "cv": args.cv,
        "num_candidates": len(search.cv_results_["params"]),
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "test_accuracy": float(test_acc),
        "elapsed_seconds": elapsed,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
