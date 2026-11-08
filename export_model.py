# export_model.py - takes the "champion" model from MLflow and saves it as champion.pkl
#
# Usage:  python export_model.py            -> exports the version with alias "champion"
#         python export_model.py --promote  -> first gives "champion" to the version with
#                                              the lowest MAE, then exports it
#                                              (same as doing it by hand in the MLflow UI)

import sys
import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

MODEL_NAME = "house-price-model"

# Use the same register (mlflow.db) that train.py writes to
mlflow.set_tracking_uri("sqlite:///mlflow.db")

if "--promote" in sys.argv:
    client = MlflowClient()
    best = None
    for version in client.search_model_versions(f"name='{MODEL_NAME}'"):
        mae = client.get_run(version.run_id).data.metrics.get("mae")
        if mae is None:
            continue
        # Lowest MAE wins; on a tie, the newest version wins
        key = (mae, -int(version.version))
        if best is None or key < best[0]:
            best = (key, version)
    if best is None:
        sys.exit("No model versions with an 'mae' metric found. Run python train.py first.")
    client.set_registered_model_alias(MODEL_NAME, "champion", best[1].version)
    print(f"Alias 'champion' -> version {best[1].version} (MAE {best[0][0]:.2f})")

# "models:/<name>@<alias>" means: the version that has this alias
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

joblib.dump(model, "champion.pkl")
print("Champion model saved to champion.pkl")
