# Data tests: is houses.csv safe to train on?

import pandas as pd

data = pd.read_csv("houses.csv")


def test_columns_are_correct():
    assert list(data.columns) == ["rooms", "size_sqft", "price_lakh"]


def test_no_empty_cells():
    assert data.notna().all().all()


def test_all_values_are_positive():
    assert (data > 0).all().all()


def test_enough_houses():
    assert len(data) >= 20


def test_rooms_are_whole_numbers():
    assert pd.api.types.is_integer_dtype(data["rooms"])


def test_bigger_houses_cost_more():
    # The pattern the model must learn: size and price go up together
    assert data["size_sqft"].corr(data["price_lakh"]) > 0.9
