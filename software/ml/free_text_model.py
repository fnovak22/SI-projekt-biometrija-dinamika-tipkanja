import json
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.feature_extractor import features_to_vector_free
from ml.fixed_text_model import get_models_dir


REQUIRED_FREE_TEXT_ENROLL_SAMPLES = 5
FREE_TEXT_ACCEPTANCE_MARGIN = -0.08
MAX_FREE_TEXT_TRAINING_SAMPLES = 25
FREE_TEXT_TRAINING_TYPES = ["free_text_enroll", "free_text_verified"]


def get_free_model_path(user_id):
    return os.path.join(get_models_dir(), f"user_{user_id}_free_model_v2.joblib")


def _training_samples(user_id, TypingSample):
    """Vraća uzorke koji smiju trenirati free-text model.

    Početnih 5 free_text_enroll uzoraka služi kao inicijalni profil. Nakon toga
    se profil smije prilagođavati samo uzorcima koje je model prihvatio
    (free_text_verified). Sumnjivi i logout uzorci se namjerno ne koriste da se
    profil ne kontaminira tuđim načinom tipkanja.
    """
    recent = TypingSample.query.filter(
        TypingSample.user_id == user_id,
        TypingSample.sample_type.in_(FREE_TEXT_TRAINING_TYPES)
    ).order_by(TypingSample.created_at.desc()).limit(MAX_FREE_TEXT_TRAINING_SAMPLES).all()
    return list(reversed(recent))


def train_free_text_model(user_id, TypingSample):
    enroll_count = TypingSample.query.filter_by(
        user_id=user_id,
        sample_type="free_text_enroll"
    ).count()

    if enroll_count < REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
        return {
            "success": False,
            "ready": False,
            "sample_count": enroll_count,
            "training_sample_count": enroll_count,
            "adaptive_sample_count": 0,
            "required_samples": REQUIRED_FREE_TEXT_ENROLL_SAMPLES,
            "message": "Još nema dovoljno prvih free-text uzoraka za treniranje."
        }

    samples = _training_samples(user_id, TypingSample)
    adaptive_count = sum(1 for sample in samples if sample.sample_type == "free_text_verified")

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
        "sample_count": enroll_count,
        "training_sample_count": len(samples),
        "adaptive_sample_count": adaptive_count,
        "max_training_samples": MAX_FREE_TEXT_TRAINING_SAMPLES,
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
    score = float(model.decision_function([vector])[0])
    accepted = bool(prediction == 1 or score >= FREE_TEXT_ACCEPTANCE_MARGIN)
    score_delta = score - FREE_TEXT_ACCEPTANCE_MARGIN

    return {
        "accepted": accepted,
        "ready": True,
        "prediction": int(prediction),
        "score": score,
        "acceptance_margin": FREE_TEXT_ACCEPTANCE_MARGIN,
        "score_delta_to_margin": score_delta,
        "message": (
            "Free-text dinamika odgovara korisniku."
            if accepted
            else "Detektirana promjena u načinu tipkanja."
        )
    }