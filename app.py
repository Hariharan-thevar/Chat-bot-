
import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Optional: load .env locally for development (keep .env out of git)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Google Gemini client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# App setup
app = Flask(__name__)  # default template_folder = ./templates
CORS(app)
logging.basicConfig(level=logging.INFO)

# Environment & model setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_model = None

if GEMINI_API_KEY:
    if genai is None:
        logging.warning(
            "google.generativeai library not available. Install google-generativeai in requirements."
        )
    else:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            # create model handle (may raise if model name invalid)
            _model = genai.GenerativeModel("gemini-1.5-flash")
            logging.info("Gemini model initialized.")
        except Exception as e:
            logging.exception("Failed to initialize Gemini model: %s", e)
            _model = None
else:
    logging.warning(
        "GEMINI_API_KEY not set. The app will start but /chat will return an error until you set the key."
    )


SYSTEM_PROMPT = "You are a helpful assistant."


@app.route("/")
def index():
    """
    Render template normally. If Jinja fails to find the template for any reason,
    attempt an absolute-file fallback and log the path checked.
    """
    try:
        return render_template("index.html")
    except Exception as exc:
        logging.exception("render_template failed, will attempt absolute-file fallback: %s", exc)
        # Build absolute path for templates/index.html relative to app root
        try:
            templates_dir = Path(app.root_path) / "templates"
            abs_path = templates_dir / "index.html"
            logging.info("Attempting to serve static file from: %s", str(abs_path))
            if abs_path.exists():
                return send_file(str(abs_path))
            else:
                logging.error("Fallback file not found at: %s", str(abs_path))
                return ("Template not found in container. Check that templates/index.html exists.", 500)
        except Exception as exc2:
            logging.exception("Absolute-file fallback failed: %s", exc2)
            return ("Internal error - check logs", 500)


@app.route("/chat", methods=["POST"])
def chat():
    """
    POST JSON: { "message": "<text>", "history": [ {"role":"user"/"assistant","content":"..."} ] }
    Returns: { "reply": "<assistant reply>" } or { "error": "..." }
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "missing 'message' field"}), 400

    if _model is None:
        return jsonify({"error": "Gemini model not configured. Set GEMINI_API_KEY in environment."}), 500

    # Build Gemini messages shape
    messages = [{"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}]
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        messages.append({"role": role, "parts": [{"text": content}]})
    messages.append({"role": "user", "parts": [{"text": message}]})

    try:
        resp = _model.generate_content(messages=messages)
        reply = getattr(resp, "text", None)
        if reply is None:
            # Fallback to string representation if .text is not available in this client version
            reply = str(resp)
        return jsonify({"reply": reply})
    except Exception as e:
        logging.exception("Gemini API call failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/_ls")
def _ls():
    """
    Debug endpoint to list files in the container (useful for Render diagnostic).
    Remove this or protect it in production.
    """
    root = Path.cwd()
    root_files = [p.name for p in root.iterdir() if p.is_file()]
    root_dirs = [p.name for p in root.iterdir() if p.is_dir()]
    templates_files = []
    tdir = root / "templates"
    if tdir.exists() and tdir.is_dir():
        templates_files = [p.name for p in tdir.iterdir() if p.is_file()]
    return jsonify(
        {
            "cwd": str(root),
            "root_files": root_files,
            "root_dirs": root_dirs,
            "templates_files": templates_files,
        }
    )


if __name__ == "__main__":
    # Bind to the port Render provides, default 5000 locally
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
