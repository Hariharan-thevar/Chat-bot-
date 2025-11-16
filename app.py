# app.py
import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Optional: load .env locally for development
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try import google generative client
try:
    import google.generativeai as genai
except Exception:
    genai = None

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_model = None

# ------------ MODEL MUST BE gemini-pro -------------
MODEL_NAME = "gemini-pro"
# ----------------------------------------------------

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        try:
            _model = genai.GenerativeModel(MODEL_NAME)
            logging.info(f"Gemini model initialized: {MODEL_NAME}")
        except Exception:
            logging.exception("Could not create GenerativeModel handle.")
            _model = None
    except Exception:
        logging.exception("Failed to configure genai API key.")
else:
    logging.warning("GEMINI_API_KEY not set or google.generativeai missing.")


SYSTEM_PROMPT = "You are a helpful assistant."


@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as exc:
        logging.exception("Jinja template error: %s", exc)
        try:
            abs_path = Path(app.root_path) / "templates/index.html"
            if abs_path.exists():
                return send_file(str(abs_path))
            return ("index.html not found inside container", 500)
        except:
            return ("Template load error", 500)


def build_prompt(messages):
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        text = ""
        if "parts" in m:
            text = m["parts"][0].get("text", "")
        else:
            text = m.get("content", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if GEMINI_API_KEY is None:
        return jsonify({"error": "GEMINI_API_KEY missing"}), 500

    messages = [{"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}]
    for h in history:
        messages.append({
            "role": h.get("role", "user"),
            "parts": [{"text": h.get("content", "")}]
        })
    messages.append({"role": "user", "parts": [{"text": message}]})

    prompt = build_prompt(messages)

    # -------- WORKING GOOGLE API CALL --------
    try:
        response = genai.generate_text(
            model=MODEL_NAME,
            prompt=prompt
        )

        reply = getattr(response, "text", None) or getattr(response, "result", None)

        if not reply:
            reply = "No response from model."

        return jsonify({"reply": reply})

    except Exception as e:
        logging.exception("Gemini error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/_ls")
def _ls():
    root = Path.cwd()
    files = [p.name for p in root.iterdir()]
    tdir = Path("templates")
    tfiles = [p.name for p in tdir.iterdir()] if tdir.exists() else []
    return jsonify({
        "root_files": files,
        "templates": tfiles
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    
