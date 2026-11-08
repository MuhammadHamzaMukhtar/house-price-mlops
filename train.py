# train.py - teaches a model to guess house prices from houses.csv,
# and records every training run in MLflow (our "result register").
#
# Usage:  python train.py          -> trains a LinearRegression model
#         python train.py tree     -> trains a DecisionTree model (to compare)
#         python train.py forest   -> trains a RandomForest model (to compare)

import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

MODEL_TYPES = ["linear", "tree", "forest"]

# MLflow setup: keep all records in one file (mlflow.db) in this folder,
# grouped under one experiment name.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("house-price-project")

# Which kind of model to train (chosen from the command line)
model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"
if model_type not in MODEL_TYPES:
    sys.exit(f"Unknown model type '{model_type}'. Choose one of: {', '.join(MODEL_TYPES)}")
test_size = 0.2

# 1. Load the data (the table of sold houses)
data = pd.read_csv("houses.csv")
print("Loaded", len(data), "houses")

# 2. Separate features (inputs) from the label (the answer)
X = data[["rooms", "size_sqft"]]   # features
y = data["price_lakh"]             # label

# 3. Split: 80% for training (practice), 20% for testing (the exam).
#    random_state=42 makes the split the same every run, so results are repeatable.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
print("Training on", len(X_train), "houses, testing on", len(X_test), "houses")

# Everything inside this block is recorded as ONE run in MLflow
with mlflow.start_run():

    # 4. Train: the model looks at the training houses and finds the pattern
    if model_type == "tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "forest":
        # 100 trees, each trained on a slightly different sample; their answers are averaged
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Exam: the model guesses prices for houses it has never seen
    guesses = model.predict(X_test)
    for (_, house), real, guess in zip(X_test.iterrows(), y_test, guesses):
        print(f"  {house['rooms']} rooms, {house['size_sqft']} sqft  ->  "
              f"real {real}  guess {guess:.1f}  off by {abs(real - guess):.1f}")

    # 6. Score: average mistake (MAE). Smaller is better.
    mae = mean_absolute_error(y_test, guesses)
    print(f"Model: {model_type}  |  MAE (average mistake): {mae:.2f} lakh")

    # MLflow: write down the settings (parameters) and the score (metric)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))
    mlflow.log_metric("mae", mae)

    # Save the score to a file so the CI quality gate can read it
    with open("metrics.json", "w") as f:
        json.dump({"mae": mae, "model_type": model_type, "rows_in_dataset": len(data)}, f)

    # MLflow: keep a copy of the model and register it as a new version.
    # MLflow blocks saving tree models unless we say we trust them -
    # we built this model ourselves, so it is safe.
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="house-price-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # 7. Also save a local copy for predict.py
    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl and recorded in MLflow")
