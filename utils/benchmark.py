import pandas as pd
from settings import (
    POSITIVE_LABEL,
    NEGATIVE_LABEL,
    MODEL_PREFIX,
    RESULTS_PATH,
    DATA_COLUMN_NAME,
    LABEL_COLUMN_NAME,
)


def calculate_match_rate(df, label):
    total = df[df[LABEL_COLUMN_NAME] == label].shape[0]
    matches = df[
        (df[LABEL_COLUMN_NAME] == label) & (df["prediction"] == f"[{label}]")
    ].shape[0]
    print(f"Total of {label} {total} matches {matches}")
    match_rate = matches / total if total > 0 else 0
    return match_rate


def test_model_accuracy(custom_results_path=None):
    df = pd.read_csv(
        f"{custom_results_path or RESULTS_PATH}{MODEL_PREFIX}_predictions.csv"
    )

    positive_rate = calculate_match_rate(df, 1)
    negative_rate = calculate_match_rate(df, 0)

    print(f"{POSITIVE_LABEL} match rate: {positive_rate:.2%}")
    print(f"{NEGATIVE_LABEL} match rate: {negative_rate:.2%}")
