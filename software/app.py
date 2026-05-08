import csv
import io
import json
import os
import random
from datetime import datetime
from functools import wraps

from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

from ml.fixed_text_model import train_fixed_model, verify_fixed_typing
from ml.feature_extractor import features_to_vector_fixed

# --- setup ---
db = SQLAlchemy()

APP_NAME = "MyNotes"

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
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


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
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


class Note(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    title = db.Column(db.String(160), nullable=False, default="Nova bilješka")
    body = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


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
            flash("Ova stranica je dostupna samo research adminu.", "danger")
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)

    return wrapper


def pending_registration_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("pending_registration_user_id"):
            flash("Prvo unesite podatke za registraciju.", "warning")
            return redirect(url_for("register"))
        return view_func(*args, **kwargs)

    return wrapper


def pending_login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
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
            "nu": 0.15,
            "accepted_when": "prediction == 1, odnosno decision_function score >= 0",
            "note": "nu=0.15 je okvirna tolerancija: model smije tretirati dio trening uzoraka kao rubne/outlier uzorke, ali nije stroga postotna granica za svaki login.",
            "fixed_text_ignored_feature": "backspace_count",
        },
        "overall_summary": summarize_vectors(vectors),
        "per_prompt_summary": prompt_stats,
    }


def choose_different_verify_prompt(current_prompt_id=None):
    candidates = [p for p in ENROLLMENT_PROMPTS if p["id"] != current_prompt_id]
    if not candidates:
        candidates = ENROLLMENT_PROMPTS
    return random.choice(candidates)


def user_is_admin(user):
    return user.role == "admin"


def extract_features(events):
    """Pretvara raw key evente u osnovne značajke za ML tim."""
    keydowns = [e for e in events if e.get("type") == "keydown" and not e.get("repeat")]
    keyups = [e for e in events if e.get("type") == "keyup"]

    pending_down = {}
    dwell_times = []
    for e in events:
        key = e.get("key")
        code = e.get("code")
        t = float(e.get("t", 0))
        k = f"{key}|{code}"
        if e.get("type") == "keydown" and not e.get("repeat"):
            pending_down.setdefault(k, []).append(t)
        elif e.get("type") == "keyup" and pending_down.get(k):
            down_t = pending_down[k].pop(0)
            dwell_times.append(round(t - down_t, 3))

    dd_intervals = []
    for i in range(1, len(keydowns)):
        dd_intervals.append(round(float(keydowns[i].get("t", 0)) - float(keydowns[i - 1].get("t", 0)), 3))

    typed_chars = [e.get("key") for e in keydowns if len(str(e.get("key", ""))) == 1]

    def avg(values):
        return round(sum(values) / len(values), 3) if values else None

    duration_ms = 0
    if len(events) >= 2:
        duration_ms = round(float(events[-1].get("t", 0)) - float(events[0].get("t", 0)), 3)

    return {
        "duration_ms": duration_ms,
        "keydown_count": len(keydowns),
        "keyup_count": len(keyups),
        "char_count": len(typed_chars),
        "backspace_count": sum(1 for e in keydowns if e.get("key") == "Backspace"),
        "avg_dwell_ms": avg(dwell_times),
        "avg_dd_interval_ms": avg(dd_intervals),
        "dwell_times_ms": dwell_times,
        "dd_intervals_ms": dd_intervals,
    }


def save_typing_sample(user, sample_type, prompt_id, prompt_text, typed_text, events):
    features = extract_features(events)
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


# --- app factory ---
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")

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

    @app.context_processor
    def inject_globals():
        return {
            "app_name": APP_NAME,
            "logged_in_username": session.get("username"),
            "logged_in_role": session.get("role"),
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
        free_text_count = TypingSample.query.filter_by(user_id=user_id, sample_type="free_text_note").count()
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
            flash("Podaci su spremljeni. Registracija završava tek nakon 20 uzoraka tipkanja.", "info")
            return redirect(url_for("register_enroll"))

        return render_template("register.html", username_value="", required_samples=REQUIRED_ENROLLMENT_SAMPLES)

    @app.get("/register/enroll")
    @pending_registration_required
    def register_enroll():
        user = User.query.get_or_404(session["pending_registration_user_id"])
        count = enrollment_count_for_user(user.id)
        if user.enrollment_complete and count >= REQUIRED_ENROLLMENT_SAMPLES:
            flash("Enrollment je već završen. Možete se prijaviti.", "success")
            session.clear()
            return redirect(url_for("login"))
        return render_template(
            "register_enroll.html",
            username=user.username,
            sequence=make_enrollment_sequence(),
            saved_count=count,
            required_samples=REQUIRED_ENROLLMENT_SAMPLES,
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
            "train_result": train_result if complete else None,
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
        session.clear()
        flash("Registracija je uspješno završena. Sada se prijavite.", "success")
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
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
                    flash("Prijavljeni ste kao research admin.", "success")
                    return redirect(url_for("admin_dashboard"))

                if not user.enrollment_complete or enrollment_count_for_user(user.id) < REQUIRED_ENROLLMENT_SAMPLES:
                    session["pending_registration_user_id"] = user.id
                    session["pending_registration_username"] = user.username
                    flash("Račun postoji, ali enrollment nije dovršen. Dovršite 20 uzoraka prije prijave.", "warning")
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
            "new_prompt": next_prompt,
            "message": "Provjera nije prošla. Dodijeljena je nova random fraza.",
        }), 403

    @app.get("/logout")
    def logout():
        session.clear()
        flash("Odjavljeni ste.", "info")
        return redirect(url_for("login"))

    @app.route("/notes/new", methods=["GET", "POST"])
    @login_required
    def note_new():
        if request.method == "POST":
            title = request.form.get("title", "").strip() or "Nova bilješka"
            body = request.form.get("body", "")
            note = Note(user_id=session["user_id"], title=title[:160], body=body, updated_at=datetime.utcnow())
            db.session.add(note)
            db.session.commit()
            flash("Bilješka je spremljena.", "success")
            return redirect(url_for("note_edit", note_id=note.id))
        return render_template("note_edit.html", note=None)

    @app.route("/notes/<int:note_id>", methods=["GET", "POST"])
    @login_required
    def note_edit(note_id):
        note = Note.query.filter_by(id=note_id, user_id=session["user_id"]).first_or_404()
        if request.method == "POST":
            note.title = (request.form.get("title", "").strip() or "Nova bilješka")[:160]
            note.body = request.form.get("body", "")
            note.updated_at = datetime.utcnow()
            db.session.commit()
            flash("Bilješka je spremljena.", "success")
            return redirect(url_for("note_edit", note_id=note.id))
        return render_template("note_edit.html", note=note)

    @app.post("/notes/<int:note_id>/delete")
    @login_required
    def note_delete(note_id):
        note = Note.query.filter_by(id=note_id, user_id=session["user_id"]).first_or_404()
        db.session.delete(note)
        db.session.commit()
        flash("Bilješka je obrisana.", "info")
        return redirect(url_for("index"))

    @app.post("/api/notes/typing-window")
    @login_required
    def api_note_typing_window():
        data = request.get_json(silent=True) or {}
        typed_text, events, error = validate_free_text_payload(data)
        if error:
            error_name, status_code = error
            return jsonify({"ok": False, "error": error_name}), status_code
        user = User.query.get_or_404(session["user_id"])
        sample, features = save_typing_sample(user, "free_text_note", "free_text", "Slobodni tekst iz bilješke", typed_text, events)
        return jsonify({"ok": True, "sample_id": sample.id, "features": features})

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
        return jsonify({"ok": True, "sample_id": sample.id, "features": features})

    @app.get("/samples")
    @login_required
    def samples():
        recent_samples = (
            TypingSample.query.filter_by(user_id=session["user_id"])
            .order_by(TypingSample.created_at.desc())
            .limit(50)
            .all()
        )
        rows = []
        for s in recent_samples:
            features = json.loads(s.features_json)
            rows.append({"sample": s, "features": features})
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
                "verify": TypingSample.query.filter_by(user_id=u.id, sample_type="verify_attempt").count(),
                "free_text": TypingSample.query.filter_by(user_id=u.id, sample_type="free_text_note").count(),
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
