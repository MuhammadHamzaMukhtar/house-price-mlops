# predict.py - uses the saved model to price a NEW house (no training here)
#
# Usage:  python predict.py              -> 3 rooms, 1350 sqft
#         python predict.py 5 2300       -> 5 rooms, 2300 sqft

import sys
import pandas as pd
import joblib

# Load the model file that train.py created
model = joblib.load("model.pkl")

# A new house the model has never seen (or the one given on the command line)
rooms = int(sys.argv[1]) if len(sys.argv) > 1 else 3
size_sqft = float(sys.argv[2]) if len(sys.argv) > 2 else 1350
new_house = pd.DataFrame({"rooms": [rooms], "size_sqft": [size_sqft]})

price = model.predict(new_house)[0]
print(f"{rooms} rooms, {size_sqft:.0f} sqft -> predicted price: {price:.1f} lakh")
