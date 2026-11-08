# Quality gate: the new model must pass the "exam" before it can be released.
# Needs metrics.json and model.pkl, so run `python train.py` first.

import json

import joblib
import pandas as pd

PASS_MARK = 3.0  # maximum allowed MAE, in lakh


def test_mae_is_below_pass_mark():
    with open("metrics.json") as f:
        mae = json.load(f)["mae"]
    assert mae < PASS_MARK, f"MAE {mae:.2f} is above the pass mark {PASS_MARK}"


def test_bigger_house_gets_higher_price():
    # Sanity check: a model that prices a big house below a small one is broken
    model = joblib.load("model.pkl")
    houses = pd.DataFrame({"rooms": [2, 5], "size_sqft": [800, 2500]})
    small, big = model.predict(houses)
    assert big > small
