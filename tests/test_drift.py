# Drift tests: does check_drift.py stay quiet for normal users and alarm for a new market?

import numpy as np
import pandas as pd

from check_drift import detect_drift, main

training = pd.read_csv("houses.csv")
rng = np.random.default_rng(42)


def normal_traffic(n=50):
    sample = training.sample(n, replace=True, random_state=1)
    return pd.DataFrame({
        "rooms": sample["rooms"].values,
        "size_sqft": (sample["size_sqft"] * rng.uniform(0.95, 1.05, n)).round().values,
    })


def luxury_traffic(n=50):
    return pd.DataFrame({
        "rooms": rng.integers(5, 9, n),
        "size_sqft": rng.integers(4000, 8001, n),
    })


def test_no_drift_for_normal_users():
    results = detect_drift(training, normal_traffic())
    assert not any(drifted for _, drifted in results.values())


def test_drift_for_luxury_market():
    results = detect_drift(training, luxury_traffic())
    assert results["size_sqft"][1] is True
    assert results["size_sqft"][0] < 0.05


def test_exit_code_is_1_on_drift(tmp_path):
    live_file = tmp_path / "predictions.csv"
    luxury_traffic().to_csv(live_file, index=False)
    assert main(live_file=live_file) == 1


def test_exit_code_is_0_without_enough_data(tmp_path):
    live_file = tmp_path / "predictions.csv"
    luxury_traffic(5).to_csv(live_file, index=False)
    assert main(live_file=live_file) == 0
