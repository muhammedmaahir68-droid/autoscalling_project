from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
import json
import joblib
import os
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(APP_DIR), "dataset.csv")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@csrf_exempt
def predict(request):

    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)

        # More specific error handling for missing keys
        if "users" not in data or "cpu" not in data:
            return JsonResponse({"error": "Missing 'users' or 'cpu' key in request"}, status=400)

        users = int(data.get("users"))
        cpu = int(data.get("cpu"))

        model = load_model()
        if model is None:
            return JsonResponse(
                {"error": "Model file missing. Run: python train_model.py"},
                status=500,
            )

        feature_count = getattr(model, "n_features_in_", 1)
        model_input = [[cpu]] if feature_count == 1 else [[users, cpu]]

        # Make prediction
        predicted_instances = int(round(float(model.predict(model_input)[0])))
        predicted_instances = max(1, predicted_instances)
        action = f"Scale to {predicted_instances} instance(s)"

        return JsonResponse({"action": action, "instances": predicted_instances})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@require_GET
def dataset(request):
    if not os.path.exists(DATASET_PATH):
        return JsonResponse({"error": "dataset.csv not found"}, status=404)

    try:
        frame = pd.read_csv(DATASET_PATH)
        if "cpu" not in frame.columns or "instances" not in frame.columns:
            return JsonResponse(
                {"error": "dataset.csv must contain 'cpu' and 'instances' columns"},
                status=400,
            )

        points = [
            {"cpu": float(row["cpu"]), "instances": float(row["instances"])}
            for _, row in frame.iterrows()
        ]

        return JsonResponse(
            {
                "points": points,
                "count": len(points),
                "cpuMin": float(frame["cpu"].min()),
                "cpuMax": float(frame["cpu"].max()),
                "instanceMin": float(frame["instances"].min()),
                "instanceMax": float(frame["instances"].max()),
            }
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)
