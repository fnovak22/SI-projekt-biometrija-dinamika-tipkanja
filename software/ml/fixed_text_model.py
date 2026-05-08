import json
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.feature_extractor import features_to_vector_fixed


def get_models_dir():
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    models_dir = os.path.join(base_dir, "instance", "models")
    os.makedirs(models_dir, exist_ok=True)

    return models_dir


def get_fixed_model_path(user_id):
    return os.path.join(
        get_models_dir(),
        f"user_{user_id}_fixed_model.joblib"
    )


def train_fixed_model(user_id, TypingSample):
    samples = TypingSample.query.filter_by(
        user_id=user_id,
        sample_type="enroll"
    ).all()

    if len(samples) < 20:
        return {
            "success": False,
            "message": "Nema dovoljno uzoraka za treniranje modela."
        }

    X = []

    for sample in samples:
        features = json.loads(sample.features_json)
        vector = features_to_vector_fixed(features)
        X.append(vector)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.15
        ))
    ])

    model.fit(X)

    model_path = get_fixed_model_path(user_id)
    joblib.dump(model, model_path)

    return {
        "success": True,
        "message": "Model uspješno istreniran."
    }


def verify_fixed_typing(user_id, features):
    model_path = get_fixed_model_path(user_id)

    if not os.path.exists(model_path):
        return {
            "accepted": False,
            "message": "Model ne postoji."
        }

    model = joblib.load(model_path)

    vector = features_to_vector_fixed(features)

    prediction = model.predict([vector])[0]
    score = model.decision_function([vector])[0]

    return {
        "accepted": bool(prediction == 1),
        "score": float(score),
        "message": (
            "Autentifikacija uspješna."
            if prediction == 1
            else "Dinamika tipkanja ne odgovara korisniku."
        )
    }