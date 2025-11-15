from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import logging

# Optional: load .env when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import google.generativeai as genai

app = Flask(__name__)  # default templates folder is ./templates
CORS(app)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not set. Set it in environment variables (Render/GitHub Secrets/.env). The app will start but calls to Gemini will fail until the key is provided.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Try to construct model only if key present; otherwise keep model None
model = None
if GEMINI_API_KEY:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logging.exception("Failed to initialize Gemini model: %s", e)
        model = None

SYSTEM_PROMPT = "You are a helpful assistant."

@app.route("/")
def index():
    # Prefer rendering template; fallback to sending static file if Jinja lookup fails
    try:
        return render_template("index.html")
    except Exception as e:
        logging.exception("render_template failed, serving static file fallback: %s", e)
        return send_from_directory("templates", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "missing 'message' field"}), 400

    user_msg = data["message"]
    history = data.get("history", [])

    if model is None:
        return jsonify({"error": "Gemini model not configured. Set GEMINI_API_KEY in environment."}), 500

    # Build messages in Gemini's expected shape
    messages = [{"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}]
    for h in history:
        messages.append({"role": h.get("role", "user"), "parts": [{"text": h.get("content", "")} ]})
    messages.append({"role": "user", "parts": [{"text": user_msg}]})

    try:
        response = model.generate_content(messages=messages)
        reply = getattr(response, "text", None)
        if reply is None:
            reply = str(response)
        return jsonify({"reply": reply})
    except Exception as e:
        logging.exception("Gemini API call failed: %s", e)
        return jsonify({"error": str(e)}), 500

# Optional debug endpoint (safe to remove)
@app.route("/_ls")
def _ls():
    root = os.getcwd()
    root_files = os.listdir(root)
    templates_files = []
    if os.path.isdir(os.path.join(root, "templates")):
        templates_files = os.listdir(os.path.join(root, "templates"))
    return jsonify({
        "cwd": root,
        "root_files": root_files,
        "templates_files": templates_files
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
