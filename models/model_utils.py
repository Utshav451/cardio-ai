#Load pre-trained pkl files, predict, ensemble.
import os, pickle
import numpy as np

MODEL_DIR = "models"

FEATURE_COLUMNS = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest":       "random_forest.pkl",
    "SVM":                 "svm.pkl",
    "KNN":                 "knn.pkl",
    "Gradient Boosting":   "gradient_boosting.pkl",
    "Naive Bayes":         "naive_bayes.pkl",
    "Neural Network":      "neural_network.pkl",
}

MODEL_COLORS = {
    "Logistic Regression": "#4F8EF7",
    "Random Forest":       "#34D399",
    "SVM":                 "#F59E0B",
    "KNN":                 "#A78BFA",
    "Gradient Boosting":   "#F87171",
    "Naive Bayes":         "#38BDF8",
    "Neural Network":      "#8A0991",
}


def load_all_models(model_dir: str = MODEL_DIR) -> dict:
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(model_dir, fname)
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def load_scaler(model_dir: str = MODEL_DIR):
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        return pickle.load(f)


def load_metrics(model_dir: str = MODEL_DIR) -> dict:
    path = os.path.join(model_dir, "metrics.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_single(input_dict: dict, models: dict, scaler) -> dict:
    row = np.array([[input_dict[col] for col in FEATURE_COLUMNS]])
    row_scaled = scaler.transform(row)
    predictions = {}
    for name, model in models.items():
        pred = int(model.predict(row_scaled)[0])
        prob = round(float(model.predict_proba(row_scaled)[0][1]) * 100, 1)
        predictions[name] = {"prediction": pred, "probability": prob}
    return predictions


def ensemble_prediction(predictions: dict) -> dict:
    votes         = [p["prediction"] for p in predictions.values()]
    probs         = [p["probability"] for p in predictions.values()]
    disease_votes = sum(votes)
    total         = len(votes)
    avg_prob      = round(sum(probs) / total, 1)

    if avg_prob < 35:
        risk_label = "Low"
        verdict    = "Not Detected"
    elif avg_prob < 60:
        risk_label = "Mild"
        verdict    = "Not Detected"
    else:
        risk_label = "High"
        verdict    = "Cardiovascular Disease Detected"

    return {
        "disease_votes":   disease_votes,
        "healthy_votes":   total - disease_votes,
        "total_models":    total,
        "avg_probability": avg_prob,
        "verdict":         verdict,
        "risk_label":      risk_label,
    }