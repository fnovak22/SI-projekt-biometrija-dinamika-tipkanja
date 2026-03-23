import os
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash

# --- setup ---
db = SQLAlchemy()

# --- models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# --- forms ---
class RegistrationForm(FlaskForm):
    username = StringField("Korisničko ime", validators=[DataRequired(), Length(min=3, max=80)])
    password = PasswordField("Lozinka", validators=[DataRequired(), Length(min=8, max=128)])
    confirm_password = PasswordField("Ponovi lozinku", validators=[DataRequired(), EqualTo("password")])

    def validate_username(self, field):
        existing = User.query.filter_by(username=field.data.strip().lower()).first()
        if existing:
            raise ValidationError("Korisničko ime je već zauzeto.")

class LoginForm(FlaskForm):
    username = StringField("Korisničko ime", validators=[DataRequired()])
    password = PasswordField("Lozinka", validators=[DataRequired()])

# --- app factory ---
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-change-me")
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../instance/app.db")
    db_uri = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    db.init_app(app)
    with app.app_context():
        db.create_all()

    # --- routes ---
    @app.get("/")
    def index():
        if session.get("user_id"):
            return f"Ulogirani korisnik: {session['username']} <br><a href='/logout'>Odjavi se</a>"
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        form = RegistrationForm()
        if form.validate_on_submit():
            username = form.username.data.strip().lower()
            password_hash = generate_password_hash(form.password.data)
            user = User(username=username, password_hash=password_hash)
            db.session.add(user)
            db.session.commit()
            flash("Registracija uspješna! Sada se prijavite.", "success")
            return redirect(url_for("login"))
        return render_template("register.html", form=form)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        form = LoginForm()
        if form.validate_on_submit():
            username = form.username.data.strip().lower()
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, form.password.data):
                session["user_id"] = user.id
                session["username"] = user.username
                flash("Prijava uspješna!", "success")
                return redirect(url_for("index"))
            else:
                flash("Krivo korisničko ime ili lozinka.", "danger")
        return render_template("login.html", form=form)

    @app.get("/logout")
    def logout():
        session.clear()
        flash("Odjavljeni ste.", "info")
        return redirect(url_for("login"))

    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)