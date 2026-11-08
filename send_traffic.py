# send_traffic.py - pretends to be 50 users calling the API
#
# Usage:  python send_traffic.py normal          -> houses like the ones the model learned from
#         python send_traffic.py drift           -> a new market: much bigger houses
#         python send_traffic.py normal URL      -> call another address, e.g. http://127.0.0.1:8080/predict (Kubernetes)

import random
import sys

import pandas as pd
import requests

# 127.0.0.1 instead of "localhost": on some machines "localhost" adds ~2 seconds per request
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/predict"
training = pd.read_csv("houses.csv")

for i in range(50):
    if mode == "drift":
        # Luxury houses: bigger than anything in the training data
        rooms = random.randint(5, 8)
        size = random.randint(4000, 8000)
    else:
        # A house similar to a training house (size changed by up to 5%)
        house = training.sample(1).iloc[0]
        rooms = int(house["rooms"])
        size = round(house["size_sqft"] * random.uniform(0.95, 1.05))

    response = requests.post(URL, json={"rooms": rooms, "size_sqft": size}, timeout=10)
    response.raise_for_status()
    print(f"{rooms} rooms, {size} sqft -> {response.json()['price_lakh']} lakh")

print(f"\nSent 50 '{mode}' requests")
