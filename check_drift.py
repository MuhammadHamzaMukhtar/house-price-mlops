# check_drift.py - are users asking about houses like the ones the model learned from?
#
# Usage:  python check_drift.py      -> exit code 0 = no drift, 1 = DRIFT
#
# The logic lives in detect_drift() so the tests in tests/test_drift.py can check it.

import sys

import pandas as pd
from scipy.stats import ks_2samp

COLUMNS = ["rooms", "size_sqft"]
P_VALUE_LIMIT = 0.05   # below this = the two lists are clearly different = drift
RECENT = 50            # how many recent predictions to compare
MIN_ROWS = 20          # fewer than this is too little to judge


def detect_drift(training: pd.DataFrame, live: pd.DataFrame, columns=COLUMNS, limit=P_VALUE_LIMIT):
    """Return {column: (p_value, drifted)} using the KS test for each column."""
    results = {}
    for column in columns:
        # KS test: how likely is it that both lists come from the same kind of houses?
        p_value = float(ks_2samp(training[column], live[column]).pvalue)
        results[column] = (p_value, p_value < limit)
    return results


def main(training_file="houses.csv", live_file="predictions.csv") -> int:
    training = pd.read_csv(training_file)
    try:
        live = pd.read_csv(live_file).tail(RECENT)   # the most recent requests
    except FileNotFoundError:
        print(f"{live_file} not found - send some traffic first")
        return 0

    if len(live) < MIN_ROWS:
        print("Not enough predictions yet - send some traffic first")
        return 0

    results = detect_drift(training, live)
    for column, (p_value, drifted) in results.items():
        print(f"{column:10}  training avg {training[column].mean():7.0f}   "
              f"live avg {live[column].mean():7.0f}   p-value {p_value:.3f}   "
              f"{'DRIFT' if drifted else 'OK'}")

    if any(drifted for _, drifted in results.values()):
        print("\nDRIFT DETECTED: users are asking about houses the model never learned from.")
        print("Action: collect real prices for these houses, add them to the data, retrain.")
        return 1   # an error code, so pipelines and alert systems can react

    print("\nNo drift: live houses look like the training houses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
