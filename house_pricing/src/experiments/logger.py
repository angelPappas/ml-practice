import csv
import os
from datetime import datetime

CSV_PATH = "kaggle_IML/experiments.csv"

FIELDNAMES = [
    "timestamp",
    "features",
    "model",
    "hyperparameters",
    "metric",
    "metric_value",
    "notes",
]


def log_experiment(
    features: list,
    model: type,
    hyperparameters: dict,
    metric: str,
    metric_value: float,
    notes: str = "",
):
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "features": features,
                "model": model,
                "hyperparameters": str(hyperparameters),
                "metric": metric,
                "metric_value": metric_value,
                "notes": notes,
            }
        )
