import csv
import io
import json
import os
import random
import secrets
from datetime import datetime
from zoneinfo import ZoneInfo
from functools import wraps

from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

from ml.fixed_text_model import FIXED_TEXT_ACCEPTANCE_MARGIN, train_fixed_model, verify_fixed_typing
from ml.feature_extractor import (
    attach_fixed_text_features,
    extract_features as extract_raw_features,
    features_to_vector_fixed,
    features_to_vector_free,
)

from ml.free_text_model import (
    FREE_TEXT_ACCEPTANCE_MARGIN,
    MAX_FREE_TEXT_TRAINING_SAMPLES,
    REQUIRED_FREE_TEXT_ENROLL_SAMPLES,
    free_text_model_exists,
    train_free_text_model,
    verify_free_text_typing,
)

# --- setup ---
db = SQLAlchemy()

APP_NAME = "MyNotes"
CROATIA_TZ = ZoneInfo("Europe/Zagreb")


def now_croatia():
    """Vraća lokalno vrijeme za Hrvatsku kao naive datetime za SQLite prikaz."""
    return datetime.now(CROATIA_TZ).replace(tzinfo=None)

# 5 fiksnih duljih fraza x 4 ponavljanja = 20 enrollment uzoraka.
ENROLLMENT_PROMPTS = [
    {"id": "p01", "text": "sigurnost korisnika ovisi o nacinu na koji tipka tekst"},
    {"id": "p02", "text": "svaki korisnik ima prepoznatljiv ritam pritiska tipki"},
    {"id": "p03", "text": "dinamika tipkanja moze pomoci u provjeri identiteta"},
    {"id": "p04", "text": "model usporeduje vrijeme drzanja tipki i razmake izmedu njih"},
    {"id": "p05", "text": "biometrijska autentifikacija ne mora koristiti dodatni hardver"},
]
REPEATS_PER_PROMPT = 4
REQUIRED_ENROLLMENT_SAMPLES = len(ENROLLMENT_PROMPTS) * REPEATS_PER_PROMPT
PROMPT_BY_ID = {p["id"]: p["text"] for p in ENROLLMENT_PROMPTS}

FEATURE_LABELS_HR = {
    "duration_ms": "Ukupno trajanje (ms)",
    "keydown_count": "Broj pritisaka tipki",
    "keyup_count": "Broj otpuštanja tipki",
    "char_count": "Broj znakova",
    "avg_dwell_ms": "Prosječno držanje tipke (ms)",
    "avg_dd_interval_ms": "Prosječni razmak između pritisaka (ms)",
    "std_dwell_ms": "Standardna devijacija držanja tipke (ms)",
    "std_dd_interval_ms": "Standardna devijacija razmaka između pritisaka (ms)",
    "typing_speed_chars_per_sec": "Brzina tipkanja (znakova/s)",
    "pause_ratio": "Udio dužih pauza",
}


def make_enrollment_sequence():
    sequence = []
    for repeat in range(REPEATS_PER_PROMPT):
        for prompt in ENROLLMENT_PROMPTS:
            sequence.append({
                "id": prompt["id"],
                "text": prompt["text"],
                "repeat": repeat + 1,
            })
    return sequence


# --- models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    enrollment_complete = db.Column(db.Boolean, nullable=False, default=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    created_at = db.Column(db.DateTime, default=now_croatia, nullable=False)


class TypingSample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    username = db.Column(db.String(80), nullable=False, index=True)
    sample_type = db.Column(db.String(30), nullable=False, default="enroll")
    prompt_id = db.Column(db.String(30), nullable=True, index=True)
    prompt_text = db.Column(db.Text, nullable=False)
    typed_text = db.Column(db.Text, nullable=False)
    events_json = db.Column(db.Text, nullable=False)
    features_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=now_croatia, nullable=False, index=True)


class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(160), nullable=False, default="Nova bilješka")
    body = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, default=now_croatia, nullable=False)
    updated_at = db.Column(db.DateTime, default=now_croatia, nullable=False, index=True)


# --- helpers ---
def ensure_schema():
    """Mali dev-only schema patch za postojeći SQLite app.db iz starije verzije prototipa."""
    inspector = inspect(db.engine)
    table_names = inspector.get_table_names()

    if "user" in table_names:
        user_columns = {c["name"] for c in inspector.get_columns("user")}
        if "enrollment_complete" not in user_columns:
            db.session.execute(text("ALTER TABLE user ADD COLUMN enrollment_complete BOOLEAN NOT NULL DEFAULT 0"))
        if "role" not in user_columns:
            db.session.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'"))

    if "typing_sample" in table_names:
        sample_columns = {c["name"] for c in inspector.get_columns("typing_sample")}
        if "prompt_id" not in sample_columns:
            db.session.execute(text("ALTER TABLE typing_sample ADD COLUMN prompt_id VARCHAR(30)"))

    db.session.commit()




def seed_admin_user():
    """Osigurava demo research admin račun: admin / admin."""
    admin = User.query.filter_by(username="admin").first()
    if admin is None:
        admin = User(
            username="admin",
            password_hash=generate_password_hash("admin"),
            enrollment_complete=True,
            role="admin",
        )
        db.session.add(admin)
    else:
        admin.password_hash = generate_password_hash("admin")
        admin.enrollment_complete = True
        admin.role = "admin"
    db.session.commit()


def redirect_authenticated_user():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("index"))


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Prvo se prijavite.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Prvo se prijavite.", "warning")
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Ova stranica je dostupna samo administratoru.", "danger")
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)

    return wrapper


def pending_registration_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if session.get("user_id"):
            return redirect_authenticated_user()
        if not session.get("pending_registration_user_id"):
            flash("Prvo unesite podatke za registraciju.", "warning")
            return redirect(url_for("register"))
        return view_func(*args, **kwargs)

    return wrapper


def pending_login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if session.get("user_id"):
            return redirect_authenticated_user()
        if not session.get("pending_login_user_id"):
            flash("Prvo unesite korisničko ime i lozinku.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper


def current_sample_count():
    if not session.get("user_id"):
        return 0
    return TypingSample.query.filter_by(user_id=session["user_id"]).count()


def enrollment_count_for_user(user_id):
    return TypingSample.query.filter_by(user_id=user_id, sample_type="enroll").count()


def free_text_sample_count_for_user(user_id):
    return TypingSample.query.filter(
        TypingSample.user_id == user_id,
        TypingSample.sample_type.like("free_text%")
    ).count()


def note_count_for_user(user_id):
    return Note.query.filter_by(user_id=user_id).count()


def _avg(values):
    return round(sum(values) / len(values), 3) if values else 0.0


def _std(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return round(variance ** 0.5, 3)


def summarize_fixed_text_profile(user_id):
    """Vraća sažetak profila za debug u browser console.

    Fixed-text model koristi samo enrollment i uspješne login pokušaje.
    Neuspješni pokušaji se nikad ne koriste za treniranje jer mogu biti impostor.
    """
    training_samples = TypingSample.query.filter(
        TypingSample.user_id == user_id,
        TypingSample.sample_type.in_(["enroll", "verify_success"]),
    ).order_by(TypingSample.created_at.asc()).all()

    failed_attempts = TypingSample.query.filter_by(user_id=user_id, sample_type="verify_failed").count()
    successful_attempts = TypingSample.query.filter_by(user_id=user_id, sample_type="verify_success").count()

    vectors = []
    by_prompt = {}
    for sample in training_samples:
        features = json.loads(sample.features_json)
        vector = features_to_vector_fixed(features)
        vectors.append(vector)
        prompt_key = sample.prompt_id or "unknown"
        prompt_bucket = by_prompt.setdefault(prompt_key, {"count": 0, "vectors": []})
        prompt_bucket["count"] += 1
        prompt_bucket["vectors"].append(vector)

    feature_names = [
        "avg_dwell_ms",
        "avg_dd_interval_ms",
        "std_dwell_ms",
        "std_dd_interval_ms",
        "typing_speed_chars_per_sec",
        "pause_ratio",
    ]

    def summarize_vectors(items):
        if not items:
            return {}
        summary = {}
        for index, name in enumerate(feature_names):
            values = [float(row[index]) for row in items]
            summary[name] = {
                "avg": _avg(values),
                "std": _std(values),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
        return summary

    prompt_stats = {}
    for prompt_id, bucket in by_prompt.items():
        prompt_stats[prompt_id] = {
            "count": bucket["count"],
            "summary": summarize_vectors(bucket["vectors"]),
        }

    return {
        "feature_names": feature_names,
        "training_sample_types": ["enroll", "verify_success"],
        "training_sample_count": len(training_samples),
        "enrollment_sample_count": TypingSample.query.filter_by(user_id=user_id, sample_type="enroll").count(),
        "successful_verify_sample_count": successful_attempts,
        "failed_verify_sample_count": failed_attempts,
        "model": {
            "type": "One-Class SVM",
            "kernel": "rbf",
            "gamma": "scale",
            "nu": 0.05,
            "accepted_when": f"prediction == 1 ili decision_function score >= {FIXED_TEXT_ACCEPTANCE_MARGIN}",
            "note": "Development tolerancija: nu=0.05, uz dodatnu marginu scorea za rubne korisničke pokušaje.",
            "fixed_text_filter": "Model koristi fixed_text_features: završne znakove fraze bez Backspace/Delete/navigation/modifier tipki i bez obrisanih znakova. Raw events_json i dalje čuva sve za free-text ML.",
        },
        "overall_summary": summarize_vectors(vectors),
        "per_prompt_summary": prompt_stats,
    }


FIXED_FEATURE_LABELS = {
    "avg_dwell_ms": "Vrijeme držanja tipke",
    "avg_dd_interval_ms": "Razmak između pritisaka",
    "std_dwell_ms": "Standardna devijacija držanja tipke",
    "std_dd_interval_ms": "Standardna devijacija razmaka",
    "typing_speed_chars_per_sec": "Brzina tipkanja",
    "pause_ratio": "Udio dužih pauza",
}


def explain_fixed_attempt(features, model_result, profile_summary):
    """Razumljiv development prikaz usporedbe trenutnog pokušaja s profilom korisnika."""
    feature_names = profile_summary.get("feature_names", [])
    vector = features_to_vector_fixed(features)
    overall = profile_summary.get("overall_summary", {})
    rows = []

    for index, name in enumerate(feature_names):
        value = float(vector[index]) if index < len(vector) else 0.0
        stats = overall.get(name, {})
        avg = stats.get("avg")
        std = stats.get("std")
        diff = None
        z_score = None
        status = "info"
        verdict = "Nema dovoljno podataka"

        if isinstance(avg, (int, float)):
            diff = round(value - float(avg), 3)
            if isinstance(std, (int, float)) and float(std) > 0.001:
                z_score = round(abs(diff) / float(std), 2)
                if z_score <= 1.25:
                    status = "good"
                    verdict = "Dobro se podudara"
                elif z_score <= 2.5:
                    status = "warn"
                    verdict = "Rubno odstupanje"
                else:
                    status = "bad"
                    verdict = "Veće odstupanje"
            else:
                # Ako je trening bio gotovo identičan, standardna devijacija može biti 0.
                tolerance = max(abs(float(avg)) * 0.20, 5.0 if name.endswith("_ms") else 0.05)
                if abs(diff) <= tolerance:
                    status = "good"
                    verdict = "Dobro se podudara"
                else:
                    status = "bad"
                    verdict = "Veće odstupanje"

        rows.append({
            "name": name,
            "label": FIXED_FEATURE_LABELS.get(name, name),
            "attempt_value": round(value, 3),
            "profile_avg": round(float(avg), 3) if isinstance(avg, (int, float)) else None,
            "profile_std": round(float(std), 3) if isinstance(std, (int, float)) else None,
            "difference": diff,
            "z_score": z_score,
            "status": status,
            "verdict": verdict,
        })

    sorted_rows = sorted(rows, key=lambda row: {"bad": 0, "warn": 1, "info": 2, "good": 3}.get(row["status"], 4))
    failed = [row for row in sorted_rows if row["status"] == "bad"]
    warnings = [row for row in sorted_rows if row["status"] == "warn"]

    score = model_result.get("score")
    margin = FIXED_TEXT_ACCEPTANCE_MARGIN
    score_delta = None
    if isinstance(score, (int, float)):
        score_delta = round(float(score) - float(margin), 4)

    near_margin = score_delta is not None and -0.02 <= score_delta < 0

    if model_result.get("accepted"):
        summary = "Model je prihvatio pokušaj. Score je iznad granice prihvaćanja ili ga je model izravno označio kao prihvatljiv."
    elif near_margin:
        summary = "Pokušaj je vrlo blizu granice prihvaćanja, ali score je još malo ispod margine. Pokušaj ponovno normalnim tempom."
    elif failed:
        summary = "Model nije prihvatio pokušaj. Najviše odskaču: " + ", ".join(row["label"] for row in failed[:3]) + "."
    elif warnings:
        summary = "Model nije prihvatio pokušaj, ali odstupanja su uglavnom rubna. Pokušaj prepisati frazu normalnim tempom."
    else:
        summary = "Model nije prihvatio pokušaj. Provjeri score, granicu prihvaćanja i broj trening uzoraka."

    return {
        "summary": summary,
        "score": score,
        "prediction": model_result.get("prediction"),
        "accepted": bool(model_result.get("accepted")),
        "near_margin": near_margin,
        "score_delta_to_margin": score_delta,
        "training_sample_count": profile_summary.get("training_sample_count", 0),
        "enrollment_sample_count": profile_summary.get("enrollment_sample_count", 0),
        "successful_verify_sample_count": profile_summary.get("successful_verify_sample_count", 0),
        "failed_verify_sample_count": profile_summary.get("failed_verify_sample_count", 0),
        "acceptance_margin": FIXED_TEXT_ACCEPTANCE_MARGIN,
        "rows": rows,
        "top_differences": sorted_rows[:3],
    }


def summarize_enrollment_progress(user_id):
    samples = TypingSample.query.filter_by(user_id=user_id, sample_type="enroll").order_by(TypingSample.created_at.asc()).all()
    # Isti skup značajki koji fixed-text model dobiva kroz features_to_vector_fixed.
    feature_names = [
        "avg_dwell_ms",
        "avg_dd_interval_ms",
        "std_dwell_ms",
        "std_dd_interval_ms",
        "typing_speed_chars_per_sec",
        "pause_ratio",
    ]
    values_by_feature = {name: [] for name in feature_names}
    per_prompt = {}
    for sample in samples:
        features = json.loads(sample.features_json)
        fixed_features = features.get("fixed_text_features", features)
        bucket = per_prompt.setdefault(sample.prompt_id or "unknown", {"count": 0})
        bucket["count"] += 1
        for name in feature_names:
            value = fixed_features.get(name)
            if isinstance(value, (int, float)):
                values_by_feature[name].append(float(value))
    overall = {}
    for name, values in values_by_feature.items():
        overall[name] = {
            "avg": _avg(values),
            "std": _std(values),
            "min": round(min(values), 3) if values else 0.0,
            "max": round(max(values), 3) if values else 0.0,
        }
    return {
        "sample_count": len(samples),
        "required_samples": REQUIRED_ENROLLMENT_SAMPLES,
        "feature_names": feature_names,
        "fixed_text_filter_note": "Statistika prikazuje očišćene fixed-text značajke koje se koriste za treniranje i provjeru fixed-text modela.",
        "ignored_for_fixed_text_now": ["backspace_count", "Backspace", "Delete", "navigacijske tipke", "modifier tipke"],
        "overall_summary": overall,
        "per_prompt_counts": per_prompt,
    }


def summarize_free_text_readiness(user_id):
    samples = TypingSample.query.filter(
        TypingSample.user_id == user_id,
        TypingSample.sample_type.like("free_text%")
    ).order_by(TypingSample.created_at.desc()).limit(20).all()
    training_count = TypingSample.query.filter_by(
        user_id=user_id,
        sample_type="free_text_enroll"
    ).count()

    # Ako aplikacija ima spremljenih dovoljno početnih uzoraka, ali model file
    # ne postoji (npr. nakon kopiranja baze, brisanja instance/models ili novog
    # pokretanja projekta), odmah ga obnovi. Tako UI ne zapne u fazi
    # "Prikupljanje početnih uzoraka" iako korisnik već ima 5+ uzoraka.
    model_ready = free_text_model_exists(user_id)
    model_rebuilt = False
    if not model_ready and training_count >= REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
        train_result = train_free_text_model(user_id, TypingSample)
        model_ready = free_text_model_exists(user_id)
        model_rebuilt = bool(train_result.get("success"))

    remaining_training = max(REQUIRED_FREE_TEXT_ENROLL_SAMPLES - training_count, 0)
    adaptive_count = TypingSample.query.filter_by(
        user_id=user_id,
        sample_type="free_text_verified"
    ).count()
    training_window_count = min(training_count + adaptive_count, MAX_FREE_TEXT_TRAINING_SAMPLES)
    if model_ready:
        phase = "verification"
        phase_label = "Model je spreman za provjeru"
    else:
        phase = "training"
        phase_label = f"Prikupljanje početnih uzoraka ({training_count}/{REQUIRED_FREE_TEXT_ENROLL_SAMPLES})"
    return {
        "sample_type": "free_text",
        "sample_count": free_text_sample_count_for_user(user_id),
        "training_count": training_count,
        "required_training_samples": REQUIRED_FREE_TEXT_ENROLL_SAMPLES,
        "remaining_training_samples": remaining_training,
        "adaptive_sample_count": adaptive_count,
        "training_window_count": training_window_count,
        "max_training_samples": MAX_FREE_TEXT_TRAINING_SAMPLES,
        "model_ready": model_ready,
        "model_rebuilt": model_rebuilt,
        "phase": phase,
        "phase_label": phase_label,
        "stored_fields": ["typed_text", "events_json", "features_json", "created_at", "user_id"],
        "features_include": [
            "avg_dwell_ms", "avg_dd_interval_ms", "std_dwell_ms", "std_dd_interval_ms",
            "typing_speed_chars_per_sec", "pause_ratio", "correction_ratio"
        ],
        "recent_sample_ids": [s.id for s in samples],
    }


FREE_TEXT_FEATURES_FOR_UI = [
    ("avg_dwell_ms", "Vrijeme držanja tipke (ms)"),
    ("avg_dd_interval_ms", "Razmak između pritisaka (ms)"),
    ("std_dwell_ms", "Std vremena držanja tipke"),
    ("std_dd_interval_ms", "Std razmaka između pritisaka"),
    ("typing_speed_chars_per_sec", "Brzina tipkanja (znakova/s)"),
    ("pause_ratio", "Udio dužih pauza"),
    ("correction_ratio", "Udio brisanja"),
]


def normalized_free_feature_map(features):
    vector = features_to_vector_free(features or {})
    return {name: vector[index] for index, (name, _label) in enumerate(FREE_TEXT_FEATURES_FOR_UI)}


def _format_dev_number(value):
    if not isinstance(value, (int, float)):
        return "—"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def summarize_free_text_profile(user_id, current_features=None, verify_result=None, mode="training"):
    """Development sažetak za free-text.

    U fazi početnog treniranja nema usporedbe s profilom jer profil još ne postoji.
    Nakon što model postoji, prikazuje se zadnji uzorak u odnosu na prosjek i std
    početnih free-text uzoraka. Prikazuju se samo značajke iz features_to_vector_free.
    """
    training_samples = TypingSample.query.filter(
        TypingSample.user_id == user_id,
        TypingSample.sample_type.in_(["free_text_enroll", "free_text_verified"])
    ).order_by(TypingSample.created_at.desc()).limit(MAX_FREE_TEXT_TRAINING_SAMPLES).all()
    training_samples = list(reversed(training_samples))

    enroll_count = TypingSample.query.filter_by(
        user_id=user_id,
        sample_type="free_text_enroll"
    ).count()

    # Ako se development panel otvara bez novog trenutnog uzorka, prikaži
    # zadnji spremljeni free-text uzorak. Ako je zadnji uzorak još uvijek jedan
    # od početnih 5 enrollment uzoraka, prikaz ostaje u "početnom" načinu bez
    # zeleno/žute/crvene procjene. Obojana usporedba ima smisla tek od prvog
    # provjernog uzorka nakon izrade profila.
    latest_sample_type = None
    if current_features is None:
        latest_sample = TypingSample.query.filter(
            TypingSample.user_id == user_id,
            TypingSample.sample_type.like("free_text%")
        ).order_by(TypingSample.created_at.desc()).first()
        if latest_sample:
            latest_sample_type = latest_sample.sample_type
        if latest_sample and latest_sample.features_json:
            current_features = json.loads(latest_sample.features_json)
    elif mode == "verification":
        latest_sample_type = "free_text_check"
    else:
        latest_sample_type = "free_text_enroll"

    model_ready = free_text_model_exists(user_id)
    if not model_ready and enroll_count >= REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
        train_result = train_free_text_model(user_id, TypingSample)
        model_ready = free_text_model_exists(user_id)

    rows = []
    has_verification_sample = latest_sample_type in {
        "free_text_check", "free_text_verified", "free_text_suspicious", "free_text_logout_triggered"
    }
    profile_ready = (
        model_ready and
        enroll_count >= REQUIRED_FREE_TEXT_ENROLL_SAMPLES and
        has_verification_sample and
        mode != "training"
    )

    profile_values = {name: [] for name, _label in FREE_TEXT_FEATURES_FOR_UI}
    for sample in training_samples:
        features = normalized_free_feature_map(json.loads(sample.features_json))
        for name, _label in FREE_TEXT_FEATURES_FOR_UI:
            value = features.get(name)
            if isinstance(value, (int, float)):
                profile_values[name].append(float(value))

    for name, label in FREE_TEXT_FEATURES_FOR_UI:
        current = None
        if current_features:
            normalized_current = normalized_free_feature_map(current_features)
            value = normalized_current.get(name)
            if isinstance(value, (int, float)):
                current = float(value)

        values = profile_values.get(name, [])
        profile_avg = _avg(values) if values else None
        profile_std = _std(values) if values else None
        diff = None
        z_score = None
        level = "neutral"
        assessment = "Profil još ne postoji"

        if profile_ready and current is not None and isinstance(profile_avg, (int, float)):
            diff = current - float(profile_avg)
            if isinstance(profile_std, (int, float)) and profile_std > 0:
                z_score = abs(diff) / float(profile_std)
                if z_score <= 1.25:
                    level = "good"
                    assessment = f"Dobro se podudara · z={z_score:.2f}"
                elif z_score <= 2.5:
                    level = "warn"
                    assessment = f"Rubno odstupanje · z={z_score:.2f}"
                else:
                    level = "bad"
                    assessment = f"Veće odstupanje · z={z_score:.2f}"
            else:
                if abs(diff) < 0.001:
                    level = "good"
                    assessment = "Dobro se podudara"
                else:
                    level = "warn"
                    assessment = "Profil nema dovoljno varijacije"
        elif current is not None:
            if model_ready and enroll_count >= REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
                assessment = "Profil je izrađen; sljedeći duži uzorak bit će provjeren"
            else:
                assessment = "Spremljeno za početno treniranje"

        rows.append({
            "key": name,
            "label": label,
            "current": _format_dev_number(current) if current is not None else "—",
            "profile": _format_dev_number(profile_avg) if profile_ready else "—",
            "profile_std": _format_dev_number(profile_std) if profile_ready else "—",
            "diff": _format_dev_number(diff) if diff is not None else "—",
            "assessment": assessment,
            "level": level,
        })

    score = verify_result.get("score") if isinstance(verify_result, dict) else None
    has_score = isinstance(score, (int, float))
    margin = verify_result.get("acceptance_margin", FREE_TEXT_ACCEPTANCE_MARGIN) if isinstance(verify_result, dict) else FREE_TEXT_ACCEPTANCE_MARGIN
    score_delta = verify_result.get("score_delta_to_margin") if isinstance(verify_result, dict) else None
    if score_delta is None and isinstance(score, (int, float)) and isinstance(margin, (int, float)):
        score_delta = float(score) - float(margin)

    score_delta_level = "neutral"
    accepted = verify_result.get("accepted") if isinstance(verify_result, dict) else None
    near_margin = isinstance(score_delta, (int, float)) and -0.02 <= float(score_delta) < 0
    if profile_ready and isinstance(score_delta, (int, float)):
        if accepted is True or float(score_delta) >= 0:
            score_delta_level = "good"
        elif near_margin:
            score_delta_level = "warn"
        else:
            score_delta_level = "bad"

    if profile_ready:
        title = "Statistika zadnjeg free-text uzorka"
        description = "Zadnji dulji odlomak uspoređen je s free-text profilom. Profil se nakon početnih 5 uzoraka prilagođava samo prihvaćenim uzorcima. Razlika od granice = score - granica; pozitivna vrijednost znači prolaz."
    elif model_ready and enroll_count >= REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
        title = "Početni free-text profil je spreman"
        description = "Prvih 5 uzoraka je spremljeno i profil je izrađen. Obojana usporedba prikazat će se od sljedećeg duljeg uzorka, jer se tek tada novi uzorak uspoređuje s gotovim profilom."
    else:
        title = "Zadnji početni free-text uzorak"
        description = "Profil još nije izrađen, zato nema procjene podudaranja. Ove vrijednosti se spremaju za početno treniranje modela."

    return {
        "profile_ready": profile_ready,
        "mode": "verification" if profile_ready else "training",
        "title": title,
        "description": description,
        "rows": rows,
        "score": _format_dev_number(score) if isinstance(score, (int, float)) else "—",
        "acceptance_margin": _format_dev_number(margin) if isinstance(margin, (int, float)) else "—",
        "score_delta_to_margin": _format_dev_number(score_delta) if isinstance(score_delta, (int, float)) else "—",
        "has_score": has_score,
        "score_delta_level": score_delta_level,
        "score_delta_help": "score - granica",
        "accepted": accepted,
        "prediction": verify_result.get("prediction") if isinstance(verify_result, dict) else None,
    }


def choose_different_verify_prompt(current_prompt_id=None):
    candidates = [p for p in ENROLLMENT_PROMPTS if p["id"] != current_prompt_id]
    if not candidates:
        candidates = ENROLLMENT_PROMPTS
    return random.choice(candidates)


def user_is_admin(user):
    return user.role == "admin"


def extract_features(events):
    """Raw feature extraction za opću/free-text analizu.

    Za fixed-text uzorke save_typing_sample dodatno sprema fixed_text_features,
    odnosno očišćenu verziju bez Backspace/Delete/navigation/modifier tipki.
    Free-text ML tim treba koristiti raw featuree i events_json, ne fixed_text_features.
    """
    return extract_raw_features(events)


def save_typing_sample(user, sample_type, prompt_id, prompt_text, typed_text, events):
    features = extract_features(events)
    if sample_type in {"enroll", "verify_attempt", "verify_success", "verify_failed"}:
        features = attach_fixed_text_features(features, events, final_text=typed_text)
    sample = TypingSample(
        user_id=user.id,
        username=user.username,
        sample_type=sample_type,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        typed_text=typed_text,
        events_json=json.dumps(events, ensure_ascii=False),
        features_json=json.dumps(features, ensure_ascii=False),
    )
    db.session.add(sample)
    db.session.commit()
    return sample, features


def validate_typing_payload(data):
    prompt_id = str(data.get("prompt_id", ""))
    prompt_text = str(data.get("prompt_text", ""))
    typed_text = str(data.get("typed_text", ""))
    events = data.get("events", [])

    if prompt_id not in PROMPT_BY_ID:
        return None, None, None, None, ("unknown_prompt", 400)
    if PROMPT_BY_ID[prompt_id] != prompt_text:
        return None, None, None, None, ("prompt_mismatch", 400)
    if typed_text != prompt_text:
        return None, None, None, None, ("typed_text_does_not_match_prompt", 400)
    if not isinstance(events, list) or len(events) < 10:
        return None, None, None, None, ("not_enough_keyboard_events", 400)

    return prompt_id, prompt_text, typed_text, events, None


def validate_free_text_payload(data):
    typed_text = str(data.get("typed_text", ""))
    events = data.get("events", [])
    if len(typed_text.strip()) < 20:
        return None, None, ("not_enough_text", 400)
    if not isinstance(events, list) or len(events) < 10:
        return None, None, ("not_enough_keyboard_events", 400)
    return typed_text, events, None


def ensure_fixed_text_features_for_existing_samples():
    """Dodaje fixed_text_features starim enrollment/verify uzorcima ako ih nemaju.

    Ovo čini stare baze kompatibilnima s novim fixed-text pristupom.
    Free-text uzorke ne diramo: ML tim za free-text treba cijeli raw signal,
    uključujući Backspace i korekcije.
    """
    fixed_types = ["enroll", "verify_attempt", "verify_success", "verify_failed"]
    samples = TypingSample.query.filter(TypingSample.sample_type.in_(fixed_types)).all()
    changed = False
    for sample in samples:
        try:
            features = json.loads(sample.features_json or "{}")
            if isinstance(features.get("fixed_text_features"), dict):
                continue
            events = json.loads(sample.events_json or "[]")
            features = attach_fixed_text_features(features, events, final_text=sample.typed_text)
            sample.features_json = json.dumps(features, ensure_ascii=False)
            changed = True
        except Exception:
            continue
    if changed:
        db.session.commit()


# --- app factory ---
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    app.config["APP_SESSION_TOKEN"] = secrets.token_hex(32)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    instance_dir = os.path.join(project_root, "instance")
    os.makedirs(instance_dir, exist_ok=True)
    db_path = os.path.join(instance_dir, "app.db")

    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    db.init_app(app)
    with app.app_context():
        db.create_all()
        ensure_schema()
        seed_admin_user()
        ensure_fixed_text_features_for_existing_samples()

    def session_has_identity():
        return any(
            session.get(key)
            for key in ("user_id", "pending_registration_user_id", "pending_login_user_id")
        )

    @app.before_request
    def invalidate_sessions_from_previous_run():
        if session_has_identity() and session.get("_app_session_token") != app.config["APP_SESSION_TOKEN"]:
            session.clear()

    @app.after_request
    def mark_current_session(response):
        if session_has_identity():
            session["_app_session_token"] = app.config["APP_SESSION_TOKEN"]
        return response

    @app.context_processor
    def inject_globals():
        pending_registration = bool(session.get("pending_registration_user_id"))
        pending_login = bool(session.get("pending_login_user_id"))
        return {
            "app_name": APP_NAME,
            "logged_in_username": session.get("username"),
            "logged_in_role": session.get("role"),
            "pending_registration": pending_registration,
            "pending_registration_username": session.get("pending_registration_username"),
            "pending_login": pending_login,
            "pending_login_username": session.get("pending_login_username"),
            "nav_sample_count": current_sample_count(),
        }

    @app.get("/")
    def index():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        user_id = session["user_id"]
        notes = Note.query.filter_by(user_id=user_id).order_by(Note.updated_at.desc()).all()
        sample_count = current_sample_count()
        enrollment_count = enrollment_count_for_user(user_id)
        free_text_count = free_text_sample_count_for_user(user_id)
        return render_template(
            "index.html",
            username=session["username"],
            notes=notes,
            sample_count=sample_count,
            note_count=len(notes),
            free_text_count=free_text_count,
            enrollment_count=enrollment_count,
            required_enrollment_samples=REQUIRED_ENROLLMENT_SAMPLES,
            prompts=ENROLLMENT_PROMPTS,
        )

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if session.get("user_id"):
            return redirect_authenticated_user()
        if session.get("pending_registration_user_id"):
            return redirect(url_for("register_enroll"))
        if session.get("pending_login_user_id"):
            return redirect(url_for("login_verify"))

        if request.method == "POST":
            username = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            errors = []
            if len(username) < 3:
                errors.append("Korisničko ime mora imati barem 3 znaka.")
            if len(password) < 8:
                errors.append("Lozinka mora imati barem 8 znakova.")
            if password != confirm_password:
                errors.append("Lozinke se ne poklapaju.")
            if User.query.filter_by(username=username).first():
                errors.append("Korisničko ime je već zauzeto.")

            if errors:
                for error in errors:
                    flash(error, "danger")
                return render_template("register.html", username_value=username, required_samples=REQUIRED_ENROLLMENT_SAMPLES)

            # Za demo je najjednostavnije: korisničko ime "admin" dobiva research dashboard.
            role = "admin" if username == "admin" else "user"
            user = User(username=username, password_hash=generate_password_hash(password), enrollment_complete=False, role=role)
            db.session.add(user)
            db.session.commit()

            session.clear()
            session["pending_registration_user_id"] = user.id
            session["pending_registration_username"] = user.username
            return redirect(url_for("register_enroll"))

        return render_template("register.html", username_value="", required_samples=REQUIRED_ENROLLMENT_SAMPLES)

    @app.get("/register/enroll")
    @pending_registration_required
    def register_enroll():
        user = User.query.get_or_404(session["pending_registration_user_id"])
        count = enrollment_count_for_user(user.id)
        if user.enrollment_complete and count >= REQUIRED_ENROLLMENT_SAMPLES:
            return redirect(url_for("register_complete"))
        return render_template(
            "register_enroll.html",
            username=user.username,
            sequence=make_enrollment_sequence(),
            saved_count=count,
            required_samples=REQUIRED_ENROLLMENT_SAMPLES,
            enrollment_stats=summarize_enrollment_progress(user.id),
        )

    @app.post("/api/registration-sample")
    @pending_registration_required
    def api_registration_sample():
        user = User.query.get_or_404(session["pending_registration_user_id"])
        count = enrollment_count_for_user(user.id)
        if count >= REQUIRED_ENROLLMENT_SAMPLES:
            user.enrollment_complete = True
            db.session.commit()
            return jsonify({"ok": True, "complete": True, "saved_count": count})

        data = request.get_json(silent=True) or {}
        prompt_id, prompt_text, typed_text, events, error = validate_typing_payload(data)
        if error:
            error_name, status_code = error
            return jsonify({"ok": False, "error": error_name}), status_code

        expected_prompt = make_enrollment_sequence()[count]
        if prompt_id != expected_prompt["id"]:
            return jsonify({"ok": False, "error": "wrong_prompt_order"}), 400

        sample, features = save_typing_sample(user, "enroll", prompt_id, prompt_text, typed_text, events)
        count += 1
        complete = count >= REQUIRED_ENROLLMENT_SAMPLES
        train_result = None

        if complete:
            user.enrollment_complete = True
            db.session.commit()
            train_result = train_fixed_model(user.id, TypingSample)

        return jsonify({
            "ok": True,
            "sample_id": sample.id,
            "features": features,
            "saved_count": count,
            "required_samples": REQUIRED_ENROLLMENT_SAMPLES,
            "complete": complete,
            "complete_url": url_for("register_complete") if complete else None,
            "train_result": train_result if complete else None,
            "enrollment_stats": summarize_enrollment_progress(user.id),
        })

    @app.get("/register/complete")
    @pending_registration_required
    def register_complete():
        user = User.query.get_or_404(session["pending_registration_user_id"])
        count = enrollment_count_for_user(user.id)
        if count < REQUIRED_ENROLLMENT_SAMPLES:
            flash("Registracija još nije završena. Potrebno je prikupiti svih 20 uzoraka.", "warning")
            return redirect(url_for("register_enroll"))

        user.enrollment_complete = True
        db.session.commit()
        return render_template(
            "register_complete.html",
            username=user.username,
            stats=summarize_enrollment_progress(user.id),
            feature_labels=FEATURE_LABELS_HR,
        )

    @app.post("/register/finish")
    @pending_registration_required
    def register_finish():
        user = User.query.get_or_404(session["pending_registration_user_id"])
        if enrollment_count_for_user(user.id) < REQUIRED_ENROLLMENT_SAMPLES:
            flash("Registracija još nije završena. Potrebno je prikupiti svih 20 uzoraka.", "warning")
            return redirect(url_for("register_enroll"))

        user.enrollment_complete = True
        db.session.commit()
        session.clear()
        flash("Registracija je uspješno završena. Sada se prijavite.", "success")
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("user_id"):
            return redirect_authenticated_user()
        if session.get("pending_registration_user_id"):
            return redirect(url_for("register_enroll"))
        if session.get("pending_login_user_id"):
            return redirect(url_for("login_verify"))

        if request.method == "POST":
            username = request.form.get("username", "").strip().lower()
            password = request.form.get("password", "")
            user = User.query.filter_by(username=username).first()

            if user and check_password_hash(user.password_hash, password):
                session.clear()

                # Research admin je demo račun za statistike i exporte.
                # Ne prolazi kroz biometrijski enrollment/login kao obični korisnici.
                if user.role == "admin":
                    session["user_id"] = user.id
                    session["username"] = user.username
                    session["role"] = user.role
                    flash("Prijavljeni ste kao administrator.", "success")
                    return redirect(url_for("admin_dashboard"))

                if not user.enrollment_complete or enrollment_count_for_user(user.id) < REQUIRED_ENROLLMENT_SAMPLES:
                    session["pending_registration_user_id"] = user.id
                    session["pending_registration_username"] = user.username
                    flash("Račun postoji, ali registracija još nije dovršena. Dovršite svih 20 uzoraka prije prijave.", "warning")
                    return redirect(url_for("register_enroll"))

                session["pending_login_user_id"] = user.id
                session["pending_login_username"] = user.username
                verify_prompt = choose_different_verify_prompt()
                session["verify_prompt_id"] = verify_prompt["id"]
                return redirect(url_for("login_verify"))

            flash("Krivo korisničko ime ili lozinka.", "danger")
            return render_template("login.html", username_value=username)

        return render_template("login.html", username_value="")

    @app.get("/login/verify")
    @pending_login_required
    def login_verify():
        user = User.query.get_or_404(session["pending_login_user_id"])
        prompt_id = session.get("verify_prompt_id")
        if prompt_id not in PROMPT_BY_ID:
            verify_prompt = random.choice(ENROLLMENT_PROMPTS)
            session["verify_prompt_id"] = verify_prompt["id"]
        else:
            verify_prompt = {"id": prompt_id, "text": PROMPT_BY_ID[prompt_id]}
        return render_template(
            "login_verify.html",
            username=user.username,
            prompt=verify_prompt,
            debug_stats=summarize_fixed_text_profile(user.id),
        )

    @app.post("/api/login-verify")
    @pending_login_required
    def api_login_verify():
        user = User.query.get_or_404(session["pending_login_user_id"])
        data = request.get_json(silent=True) or {}
        prompt_id, prompt_text, typed_text, events, error = validate_typing_payload(data)
        if error:
            error_name, status_code = error
            return jsonify({"ok": False, "error": error_name}), status_code
        if prompt_id != session.get("verify_prompt_id"):
            return jsonify({"ok": False, "error": "wrong_verify_prompt"}), 400

        sample, features = save_typing_sample(user, "verify_attempt", prompt_id, prompt_text, typed_text, events)

        # Fixed-text model namjerno ignorira backspace_count u vektoru.
        # Model se trenira iz enrollment uzoraka + ranijih uspješnih verify pokušaja.
        # Trenutni pokušaj se NE dodaje u trening prije provjere.
        train_fixed_model(user.id, TypingSample)
        model_result = verify_fixed_typing(user.id, features)
        attempt_vector = features_to_vector_fixed(features)
        debug_stats = summarize_fixed_text_profile(user.id)
        debug_explanation = explain_fixed_attempt(features, model_result, debug_stats)

        if model_result["accepted"]:
            sample.sample_type = "verify_success"
            db.session.commit()

            # Uspješni login pokušaji su vjerojatno vlasnik računa, pa ih smijemo dodati
            # u profil za sljedeće provjere. Neuspješne pokušaje nikad ne dodajemo.
            retrain_result = train_fixed_model(user.id, TypingSample)

            session.clear()
            session["user_id"] = user.id
            session["username"] = user.username
            session["role"] = user.role
            return jsonify({
                "ok": True,
                "accepted": True,
                "redirect_url": url_for("index"),
                "sample_id": sample.id,
                "features": features,
                "fixed_vector": attempt_vector,
                "model": model_result,
                "retrain_result": retrain_result,
                "debug_stats_before_retrain": debug_stats,
                "debug_explanation": debug_explanation,
            })

        sample.sample_type = "verify_failed"
        db.session.commit()
        next_prompt = choose_different_verify_prompt(prompt_id)
        session["verify_prompt_id"] = next_prompt["id"]

        return jsonify({
            "ok": True,
            "accepted": False,
            "sample_id": sample.id,
            "features": features,
            "fixed_vector": attempt_vector,
            "model": model_result,
            "debug_stats": summarize_fixed_text_profile(user.id),
            "debug_explanation": debug_explanation,
            "new_prompt": next_prompt,
            "message": "Provjera nije prošla. Dodijeljena je nova random fraza.",
        }), 403

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/notes/new", methods=["GET", "POST"])
    @login_required
    def note_new():
        if request.method == "POST":
            title = request.form.get("title", "").strip() or "Nova bilješka"
            body = request.form.get("body", "")
            note = Note(user_id=session["user_id"], title=title[:160], body=body, updated_at=now_croatia())
            db.session.add(note)
            db.session.commit()
            flash("Bilješka je spremljena.", "success")
            return redirect(url_for("note_edit", note_id=note.id))
        return render_template(
            "note_edit.html",
            note=None,
            free_text_readiness=summarize_free_text_readiness(session.get("user_id", 0)) if session.get("user_id") else None,
            free_text_profile=summarize_free_text_profile(
                session.get("user_id", 0),
                mode="verification" if session.get("user_id") and free_text_model_exists(session.get("user_id")) else "training"
            ) if session.get("user_id") else None,
            suspicious_count=session.get("free_text_suspicious_count", 0),
        )

    @app.route("/notes/<int:note_id>", methods=["GET", "POST"])
    @login_required
    def note_edit(note_id):
        note = Note.query.filter_by(id=note_id, user_id=session["user_id"]).first_or_404()
        if request.method == "POST":
            note.title = (request.form.get("title", "").strip() or "Nova bilješka")[:160]
            note.body = request.form.get("body", "")
            note.updated_at = now_croatia()
            db.session.commit()
            flash("Bilješka je spremljena.", "success")
            return redirect(url_for("note_edit", note_id=note.id))
        return render_template(
            "note_edit.html",
            note=note,
            free_text_readiness=summarize_free_text_readiness(session["user_id"]),
            free_text_profile=summarize_free_text_profile(
                session["user_id"],
                mode="verification" if free_text_model_exists(session["user_id"]) else "training"
            ),
            suspicious_count=session.get("free_text_suspicious_count", 0),
        )

    @app.post("/notes/<int:note_id>/delete")
    @login_required
    def note_delete(note_id):
        note = Note.query.filter_by(id=note_id, user_id=session["user_id"]).first_or_404()
        db.session.delete(note)
        db.session.commit()
        flash("Bilješka je obrisana.", "info")
        return redirect(url_for("index"))
    
    def is_free_text_rule_suspicious(user_id, current_features):
        enroll_samples = TypingSample.query.filter_by(
            user_id=user_id,
            sample_type="free_text_enroll"
        ).all()

        if len(enroll_samples) < 3:
            return False, {}

        speeds = []
        intervals = []
        pauses = []
        correction_ratios = []

        for sample in enroll_samples:
            features = normalized_free_feature_map(json.loads(sample.features_json))
            speeds.append(float(features.get("typing_speed_chars_per_sec", 0)))
            intervals.append(float(features.get("avg_dd_interval_ms", 0)))
            pauses.append(float(features.get("pause_ratio", 0)))
            correction_ratios.append(float(features.get("correction_ratio", 0)))

        avg_speed = sum(speeds) / len(speeds)
        avg_interval = sum(intervals) / len(intervals)
        avg_pause = sum(pauses) / len(pauses)
        avg_correction_ratio = sum(correction_ratios) / len(correction_ratios)

        current = normalized_free_feature_map(current_features)
        current_speed = float(current.get("typing_speed_chars_per_sec", 0))
        current_interval = float(current.get("avg_dd_interval_ms", 0))
        current_pause = float(current.get("pause_ratio", 0))
        current_correction_ratio = float(current.get("correction_ratio", 0))

        suspicious = (
            current_speed < avg_speed * 0.65 or
            current_speed > avg_speed * 1.60 or
            current_interval > avg_interval * 1.60 or
            current_pause > avg_pause + 0.20 or
            current_correction_ratio > avg_correction_ratio + 0.08
        )

        details = {
            "avg_speed": avg_speed,
            "current_speed": current_speed,
            "avg_interval": avg_interval,
            "current_interval": current_interval,
            "avg_pause": avg_pause,
            "current_pause": current_pause,
            "avg_correction_ratio": avg_correction_ratio,
            "current_correction_ratio": current_correction_ratio,
        }

        return suspicious, details
    
    @app.post("/api/notes/typing-window")
    @login_required
    def api_note_typing_window():
        data = request.get_json(silent=True) or {}
        typed_text, events, error = validate_free_text_payload(data)

        if error:
            error_name, status_code = error
            return jsonify({"ok": False, "error": error_name}), status_code

        user = User.query.get_or_404(session["user_id"])

        enroll_count = TypingSample.query.filter_by(
            user_id=user.id,
            sample_type="free_text_enroll"
        ).count()

        model_exists = free_text_model_exists(user.id)

        # 1) PRVI DUGI FREE-TEXT UNOS: skuplja enrollment uzorke
        if not model_exists and enroll_count < REQUIRED_FREE_TEXT_ENROLL_SAMPLES:
            sample, features = save_typing_sample(
                user,
                "free_text_enroll",
                "free_text",
                "Prvi slobodni tekst za treniranje free-text modela",
                typed_text,
                events
            )

            train_result = train_free_text_model(user.id, TypingSample)

            return jsonify({
                "ok": True,
                "mode": "training",
                "sample_id": sample.id,
                "sample_type": sample.sample_type,
                "features": features,
                "train_result": train_result,
                "suspicious_count": 0,
                "logout_required": False,
                "free_text_readiness": summarize_free_text_readiness(user.id),
                "free_text_profile": summarize_free_text_profile(user.id, features, mode="training"),
                "message": "Spremljen je početni free-text uzorak."
            })

        # 2) AKO MODEL JOŠ NIJE NASTAO, pokušaj ga istrenirati
        if not model_exists:
            train_result = train_free_text_model(user.id, TypingSample)
            model_exists = free_text_model_exists(user.id)

            if not model_exists:
                sample, features = save_typing_sample(
                    user,
                    "free_text_enroll",
                    "free_text",
                    "Dodatni slobodni tekst za treniranje free-text modela",
                    typed_text,
                    events
                )

                return jsonify({
                    "ok": True,
                    "mode": "training",
                    "sample_id": sample.id,
                    "sample_type": sample.sample_type,
                    "features": features,
                    "train_result": train_result,
                    "suspicious_count": 0,
                    "logout_required": False,
                    "free_text_readiness": summarize_free_text_readiness(user.id),
                    "free_text_profile": summarize_free_text_profile(user.id, features, mode="training"),
                    "message": "Model još nije spreman za provjeru. Prikupljaju se početni uzorci."
                })

        # 3) SVAKI SLJEDEĆI FREE-TEXT UNOS: provjera
        sample, features = save_typing_sample(
            user,
            "free_text_check",
            "free_text",
            "Slobodni tekst za tihu provjeru korisnika",
            typed_text,
            events
        )

        verify_result = verify_free_text_typing(user.id, features)
        
        rule_suspicious, rule_details = is_free_text_rule_suspicious(user.id, features)

        # Dodatna pravila ostaju samo development signal.
        # Konačnu odluku donosi free-text model s marginom prihvaćanja,
        # kako UI statistika ne bi pokazivala zeleno dok skriveno pravilo odjavljuje korisnika.
        if rule_suspicious:
            verify_result["rule_warning"] = True
            verify_result["rule_details"] = rule_details
        else:
            verify_result["rule_warning"] = False

        suspicious_count = session.get("free_text_suspicious_count", 0)
        logout_required = False

        profile_summary = summarize_free_text_profile(user.id, features, verify_result, mode="verification")
        train_result = None

        if verify_result["accepted"] is True:
            sample.sample_type = "free_text_verified"
            suspicious_count = 0

        elif verify_result["accepted"] is False:
            suspicious_count += 1

            if suspicious_count >= 3:
                sample.sample_type = "free_text_logout_triggered"
                logout_required = True
            else:
                sample.sample_type = "free_text_suspicious"

        session["free_text_suspicious_count"] = suspicious_count
        db.session.commit()

        if verify_result["accepted"] is True:
            train_result = train_free_text_model(user.id, TypingSample)

        response = {
            "ok": True,
            "mode": "verification",
            "sample_id": sample.id,
            "sample_type": sample.sample_type,
            "features": features,
            "verify_result": verify_result,
            "suspicious_count": suspicious_count,
            "logout_required": logout_required,
            "free_text_readiness": summarize_free_text_readiness(user.id),
            "free_text_profile": profile_summary,
            "train_result": train_result,
            "message": verify_result.get("message")
        }

        if logout_required:
            session.clear()
            response["redirect_url"] = url_for("login")
            response["message"] = "Tri puta zaredom je detektirana promjena u načinu tipkanja. Korisnik je odjavljen."

        return jsonify(response)

    @app.get("/collect")
    @login_required
    def collect():
        return render_template("collect.html", prompts=ENROLLMENT_PROMPTS)

    @app.post("/api/typing-sample")
    @login_required
    def api_typing_sample():
        data = request.get_json(silent=True) or {}
        prompt_id, prompt_text, typed_text, events, error = validate_typing_payload(data)
        if error:
            error_name, status_code = error
            return jsonify({"ok": False, "error": error_name}), status_code

        sample_type = str(data.get("sample_type", "extra_enroll"))[:30]
        user = User.query.get_or_404(session["user_id"])
        sample, features = save_typing_sample(user, sample_type, prompt_id, prompt_text, typed_text, events)
        return jsonify({"ok": True, "sample_id": sample.id, "features": features, "free_text_readiness": summarize_free_text_readiness(user.id), "free_text_profile": summarize_free_text_profile(user.id, features)})

    @app.get("/samples")
    @login_required
    def samples():
        recent_samples = (
            TypingSample.query.filter_by(user_id=session["user_id"])
            .order_by(TypingSample.created_at.desc())
            .limit(50)
            .all()
        )

        label_map = {
            "enroll": ("Registracijski uzorak", "Spremljeno"),
            "extra_enroll": ("Dodatni uzorak", "Spremljeno"),
            "verify_success": ("Prijava", "Uspješno"),
            "verify_attempt": ("Prijava", "Pokušaj"),
            "verify_failed": ("Prijava", "Neuspješno"),
            "free_text_enroll": ("Pisanje bilješke", "Spremljeno"),
            "free_text_check": ("Pisanje bilješke", "Provjereno"),
            "free_text_verified": ("Pisanje bilješke", "Potvrđeno"),
            "free_text_suspicious": ("Pisanje bilješke", "Sumnjivo"),
            "free_text_logout_triggered": ("Pisanje bilješke", "Odjavljeno"),
        }

        rows = []
        for s in recent_samples:
            features = json.loads(s.features_json)
            label, result = label_map.get(s.sample_type, ("Aktivnost tipkanja", "Spremljeno"))
            duration = features.get("duration_ms")
            speed = features.get("typing_speed_chars_per_sec")
            rows.append({
                "sample": s,
                "label": label,
                "result": result,
                "duration": f"{duration:.0f} ms" if isinstance(duration, (int, float)) else "-",
                "speed": f"{speed:.2f} zn./s" if isinstance(speed, (int, float)) else "-",
            })
        return render_template("samples.html", rows=rows)

    @app.get("/admin")
    @admin_required
    def admin_dashboard():
        users = User.query.order_by(User.created_at.desc()).all()
        user_rows = []
        for u in users:
            user_rows.append({
                "user": u,
                "notes": note_count_for_user(u.id),
                "enroll": enrollment_count_for_user(u.id),
                "verify": TypingSample.query.filter(
                    TypingSample.user_id == u.id,
                    TypingSample.sample_type.in_(["verify_success", "verify_failed", "verify_attempt"])
                ).count(),
                "free_text": free_text_sample_count_for_user(u.id),
                "total": TypingSample.query.filter_by(user_id=u.id).count(),
            })
        sample_types = db.session.query(TypingSample.sample_type, func.count(TypingSample.id)).group_by(TypingSample.sample_type).all()
        recent_samples = TypingSample.query.order_by(TypingSample.created_at.desc()).limit(30).all()
        return render_template(
            "admin.html",
            user_rows=user_rows,
            sample_types=sample_types,
            recent_samples=recent_samples,
            total_users=User.query.count(),
            total_notes=Note.query.count(),
            total_samples=TypingSample.query.count(),
        )

    @app.get("/export/typing-samples.csv")
    @login_required
    def export_typing_samples_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "sample_id",
            "user_id",
            "username",
            "sample_type",
            "prompt_id",
            "created_at",
            "prompt_text",
            "typed_text",
            "duration_ms",
            "keydown_count",
            "keyup_count",
            "char_count",
            "backspace_count",
            "avg_dwell_ms",
            "avg_dd_interval_ms",
            "std_dwell_ms",
            "std_dd_interval_ms",
            "typing_speed_chars_per_sec",
            "pause_ratio",
            "dwell_times_ms",
            "dd_intervals_ms",
            "events_json",
        ])

        query = TypingSample.query
        if session.get("role") != "admin":
            query = query.filter_by(user_id=session["user_id"])

        for s in query.order_by(TypingSample.created_at.asc()).all():
            features = json.loads(s.features_json)
            writer.writerow([
                s.id,
                s.user_id,
                s.username,
                s.sample_type,
                s.prompt_id,
                s.created_at.isoformat(),
                s.prompt_text,
                s.typed_text,
                features.get("duration_ms"),
                features.get("keydown_count"),
                features.get("keyup_count"),
                features.get("char_count"),
                features.get("backspace_count"),
                features.get("avg_dwell_ms"),
                features.get("avg_dd_interval_ms"),
                features.get("std_dwell_ms"),
                features.get("std_dd_interval_ms"),
                features.get("typing_speed_chars_per_sec"),
                features.get("pause_ratio"),
                json.dumps(features.get("dwell_times_ms", [])),
                json.dumps(features.get("dd_intervals_ms", [])),
                s.events_json,
            ])

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=mynotes-typing-samples.csv"},
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
