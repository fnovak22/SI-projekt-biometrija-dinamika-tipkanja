import csv
import io
import json
import os
from datetime import datetime
from functools import wraps

from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

# --- setup ---
db = SQLAlchemy()

# Pametniji fixed-text pristup za SVM:
# 5 fiksnih duljih fraza x 4 ponavljanja = 20 enrollment uzoraka.
# Nije 20 potpuno različitih rečenica jer ML timu trebaju usporedivi uzorci po istom promptu.
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class TypingSample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    username = db.Column(db.String(80), nullable=False, index=True)
    sample_type = db.Column(db.String(20), nullable=False, default="enroll")
    prompt_id = db.Column(db.String(20), nullable=True, index=True)
    prompt_text = db.Column(db.Text, nullable=False)
    typed_text = db.Column(db.Text, nullable=False)
    events_json = db.Column(db.Text, nullable=False)
    features_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)


# --- helpers ---
def ensure_schema():
    """Mali dev-only schema patch za postojeći SQLite app.db iz starije verzije prototipa."""
    inspector = inspect(db.engine)
    table_names = inspector.get_table_names()

    if "user" in table_names:
        user_columns = {c["name"] for c in inspector.get_columns("user")}
        if "enrollment_complete" not in user_columns:
            db.session.execute(text("ALTER TABLE user ADD COLUMN enrollment_complete BOOLEAN NOT NULL DEFAULT 0"))

    if "typing_sample" in table_names:
        sample_columns = {c["name"] for c in inspector.get_columns("typing_sample")}
        if "prompt_id" not in sample_columns:
            db.session.execute(text("ALTER TABLE typing_sample ADD COLUMN prompt_id VARCHAR(20)"))

    db.session.commit()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Prvo se prijavite.", "warning")
            return redirect(url_for("login"))
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


def extract_features(events):
    """Pretvara raw key evente u osnovne značajke za ML tim."""
    keydowns = [e for e in events if e.get("type") == "keydown" and not e.get("repeat")]
    keyups = [e for e in events if e.get("type") == "keyup"]

    # Dwell / hold time: vrijeme od keydown do keyup za istu tipku.
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

    # DD interval: razmak između dva uzastopna keydowna.
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


# Privremeni placeholder dok tim za model ne spoji stvarni SVM.
def verify_typing_with_model_stub(user, features, prompt_id):
    return {
        "accepted": True,
        "score": None,
        "model_status": "placeholder_no_model_yet",
        "message": "Model još nije spojen, pa prototip trenutno propušta ispravno prepisan tekst.",
    }


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

    @app.context_processor
    def inject_globals():
        return {
            "logged_in_username": session.get("username"),
            "nav_sample_count": current_sample_count(),
        }

    # --- routes ---
    @app.get("/")
    def index():
        if not session.get("user_id"):
            return redirect(url_for("login"))
        sample_count = current_sample_count()
        enrollment_count = enrollment_count_for_user(session["user_id"])
        total_users = User.query.count()
        total_samples = TypingSample.query.count()
        return render_template(
            "index.html",
            username=session["username"],
            sample_count=sample_count,
            enrollment_count=enrollment_count,
            required_enrollment_samples=REQUIRED_ENROLLMENT_SAMPLES,
            total_users=total_users,
            total_samples=total_samples,
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

            user = User(username=username, password_hash=generate_password_hash(password), enrollment_complete=False)
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

        expected_sequence = make_enrollment_sequence()
        expected_prompt = expected_sequence[count]
        if prompt_id != expected_prompt["id"]:
            return jsonify({"ok": False, "error": "wrong_prompt_order"}), 400

        sample, features = save_typing_sample(user, "enroll", prompt_id, prompt_text, typed_text, events)
        count += 1
        complete = count >= REQUIRED_ENROLLMENT_SAMPLES
        if complete:
            user.enrollment_complete = True
            db.session.commit()

        return jsonify({
            "ok": True,
            "sample_id": sample.id,
            "features": features,
            "saved_count": count,
            "required_samples": REQUIRED_ENROLLMENT_SAMPLES,
            "complete": complete,
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
                if not user.enrollment_complete or enrollment_count_for_user(user.id) < REQUIRED_ENROLLMENT_SAMPLES:
                    session["pending_registration_user_id"] = user.id
                    session["pending_registration_username"] = user.username
                    flash("Račun postoji, ali enrollment nije dovršen. Dovršite 20 uzoraka prije prijave.", "warning")
                    return redirect(url_for("register_enroll"))

                session["pending_login_user_id"] = user.id
                session["pending_login_username"] = user.username
                return redirect(url_for("login_verify"))

            flash("Krivo korisničko ime ili lozinka.", "danger")
            return render_template("login.html", username_value=username)

        return render_template("login.html", username_value="")

    @app.get("/login/verify")
    @pending_login_required
    def login_verify():
        user = User.query.get_or_404(session["pending_login_user_id"])
        # Za demo biramo prvi prompt. Kad ML tim spoji model, može se birati random prompt po prompt_id-u.
        verify_prompt = ENROLLMENT_PROMPTS[0]
        return render_template("login_verify.html", username=user.username, prompt=verify_prompt)

    @app.post("/api/login-verify")
    @pending_login_required
    def api_login_verify():
        user = User.query.get_or_404(session["pending_login_user_id"])
        data = request.get_json(silent=True) or {}
        prompt_id, prompt_text, typed_text, events, error = validate_typing_payload(data)
        if error:
            error_name, status_code = error
            return jsonify({"ok": False, "error": error_name}), status_code

        sample, features = save_typing_sample(user, "verify_attempt", prompt_id, prompt_text, typed_text, events)
        model_result = verify_typing_with_model_stub(user, features, prompt_id)

        if model_result["accepted"]:
            session.clear()
            session["user_id"] = user.id
            session["username"] = user.username
            return jsonify({
                "ok": True,
                "accepted": True,
                "redirect_url": url_for("index"),
                "sample_id": sample.id,
                "features": features,
                "model": model_result,
            })

        return jsonify({
            "ok": True,
            "accepted": False,
            "sample_id": sample.id,
            "features": features,
            "model": model_result,
        }), 403

    @app.get("/logout")
    def logout():
        session.clear()
        flash("Odjavljeni ste.", "info")
        return redirect(url_for("login"))

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

        sample_type = str(data.get("sample_type", "extra_enroll"))[:20]
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

        for s in TypingSample.query.order_by(TypingSample.created_at.asc()).all():
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
            headers={"Content-Disposition": "attachment; filename=typing-samples.csv"},
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
