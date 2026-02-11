import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "dataset.csv")

data = pd.read_csv(file_path)

X = data[["cpu"]]
y = data["instances"]

model = LinearRegression()
model.fit(X, y)

models_dir = os.path.join(BASE_DIR, "autoscaling", "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "model.pkl")
joblib.dump(model, model_path)

print(f"Model trained and saved successfully at: {model_path}")
