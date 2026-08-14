"""
preview.py -- Standalone preview server for the DEAD frontend.
----------------------------------------------------------------
Run this to browse all HTML pages with dummy data. This does NOT touch
or depend on the main app.py or any backend AI modules.

Usage:
    python preview.py

Then open http://127.0.0.1:5050 in your browser.
"""

from flask import Flask, render_template

app = Flask(
    __name__,
    template_folder="src/frontend/templates",
    static_folder="src/frontend/static",
)


def auth_context():
    return {"logged_in": True}


def public_context():
    return {"logged_in": False}


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", **public_context())


@app.route("/login")
def login_page():
    return render_template("login.html", **public_context())


@app.route("/signup")
def signup_page():
    return render_template("signup.html", **public_context())


# ---------------------------------------------------------------------------
# Authenticated routes
# ---------------------------------------------------------------------------
@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", **auth_context())


@app.route("/beneficiaries")
def beneficiaries_page():
    return render_template("beneficiaries.html", **auth_context())


@app.route("/death-registry")
def death_registry_page():
    return render_template("death_registry.html", **auth_context())


@app.route("/verify")
def verify_page():
    return render_template("verify.html", **auth_context())


@app.route("/handover")
def handover_page():
    return render_template("handover.html", **auth_context())


@app.route("/logout")
def logout():
    return 'Logged out. <a href="/">Go to home</a>'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n  DEAD System -- Frontend Preview Server")
    print("  Open http://127.0.0.1:5050 in your browser\n")
    app.run(host="127.0.0.1", port=5050, debug=True)