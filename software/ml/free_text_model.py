import json
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.feature_extractor import features_to_vector_free
from ml.fixed_text_model import get_models_dir


REQUIRED_FREE_TEXT_ENROLL_SAMPLES = 5


def get_free_model_path(user_id):
    return os.path.join(get_models_dir(), f"user_{user_id}_free_model.joblib")


def train_free_text_model(user_id, TypingSample):
    samples = TypingSample.query.filter_by(
        user_id=user_id,
        sample_type="free_text_enroll"
    ).all()

    if len(samples) < REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
        return {
            "success": False,
            "ready": False,
            "sample_count": len(samples),
            "required_samples": REQUIRED_FREE_TEXT_ENROLL_SAMPLES,
            "message": "Još nema dovoljno prvih free-text uzoraka za treniranje."
        }

    X = []

    for sample in samples:
        features = json.loads(sample.features_json)
        vector = features_to_vector_free(features)
        X.append(vector)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", OneClassSVM(kernel="rbf", gamma="scale", nu=0.1))
    ])

    model.fit(X)

    joblib.dump(model, get_free_model_path(user_id))

    return {
        "success": True,
        "ready": True,
        "sample_count": len(samples),
        "message": "Free-text model je istreniran."
    }


def free_text_model_exists(user_id):
    return os.path.exists(get_free_model_path(user_id))


def verify_free_text_typing(user_id, features):
    model_path = get_free_model_path(user_id)

    if not os.path.exists(model_path):
        return {
            "accepted": None,
            "ready": False,
            "message": "Free-text model još nije istreniran."
        }

    model = joblib.load(model_path)
    vector = features_to_vector_free(features)

    prediction = model.predict([vector])[0]
    score = model.decision_function([vector])[0]

    return {
        "accepted": bool(prediction == 1),
        "ready": True,
        "prediction": int(prediction),
        "score": float(score),
        "message": (
            "Free-text dinamika odgovara korisniku."
            if prediction == 1
            else "Detektirana promjena u načinu tipkanja."
        )
    }