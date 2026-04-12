import json
import os
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid, train_test_split

PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}


def main():
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    params = list(ParameterGrid(PARAM_GRID))[task_id]

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    Path("array_results").mkdir(exist_ok=True)
    out = {
        "task_id": task_id,
        "params": params,
        "test_accuracy": float(acc),
        "hostname": os.uname().nodename,
    }
    Path(f"array_results/result_{task_id}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
