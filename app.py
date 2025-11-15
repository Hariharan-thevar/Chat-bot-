# app.py
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

# Try import google generative client
try:
    import google.generativeai as genai
except Exception:
    genai = None

app = Flask(__name__)  # default templates folder = ./templates
CORS(app)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_model = None

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Try to create a model handle; if this fails we'll still keep going but _model will be None
        try:
            _model = genai.GenerativeModel("gemini-1.5-flash")
            logging.info("Gemini model initialized.")
        except Exception:
            # Some client versions might not support GenerativeModel(...) or may require a different name;
            # keep _model None and rely on genai.generate_text/generate fallback below.
            logging.exception("Could not create GenerativeModel handle; will try genai-level functions instead.")
            _model = None
    except Exception:
        logging.exception("Failed to configure genai library with GEMINI_API_KEY.")
else:
    if GEMINI_API_KEY and genai is None:
        logging.warning("GEMINI_API_KEY provided but google.generativeai library is missing.")
    else:
        logging.warning("GEMINI_API_KEY not set. /chat will return an error until you set the env var.")


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


def build_prompt_from_messages(messages):
    """
    Convert messages (list of dicts in the shape used earlier) into a single prompt string.
    messages is expected to be a list of dicts like: {"role": "system/user/assistant", "parts": [{"text": "..."}]}
    or the older simple shape we used in history: {"role": "...", "content": "..."}
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        # support both shapes: parts -> text, or content
        if "parts" in m and isinstance(m["parts"], list) and len(m["parts"]) > 0:
            text = m["parts"][0].get("text", "")
        else:
            text = m.get("content", "")
        parts.append(f"{role.upper()}: {text}")
    return "\n".join(parts)


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

    if GEMINI_API_KEY is None:
        return jsonify({"error": "GEMINI_API_KEY not configured in environment."}), 500

    # Build messages in the shape we attempted before (for modern API)
    messages = [{"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}]
    for h in history:
        # Accept either {"role":"user","content":"..."} or {"role":"user","parts":[{"text":"..."}]}
        if "content" in h:
            messages.append({"role": h.get("role", "user"), "parts": [{"text": h.get("content", "")}]})
        else:
            # assume already in parts form
            messages.append({"role": h.get("role", "user"), "parts": h.get("parts", [{"text": ""}])})
    # Add user's latest message
    messages.append({"role": "user", "parts": [{"text": message}]})

    # First, try the modern method (may raise TypeError like you saw if the client signature differs)
    try:
        if _model is not None:
            try:
                resp = _model.generate_content(messages=messages)
                reply = getattr(resp, "text", None)
                if reply is None:
                    reply = str(resp)
                return jsonify({"reply": reply})
            except TypeError as te:
                # This is the exact error you saw; fall back below.
                logging.warning("generate_content(messages=...) not supported on this runtime: %s", te)
            except Exception as e:
                # Unexpected error from generate_content — surface it
                logging.exception("generate_content(messages=...) call raised an exception: %s", e)
                # We'll try fallbacks below before returning error
        else:
            logging.info("_model handle not available; will try genai-level functions.")
    except Exception:
        logging.exception("Unexpected error while calling generate_content; will try fallbacks.")

    # Build a single prompt string from messages (system + history + user) for fallback calls
    prompt = build_prompt_from_messages(messages)

    # Try several fallback calling styles (best-effort)
    last_exc = None
    # 1) try a model-level generate if available: model.generate(prompt=...)
    if _model is not None and hasattr(_model, "generate"):
        try:
            gen = getattr(_model, "generate")
            resp = gen(prompt=prompt)
            reply = getattr(resp, "text", None) or str(resp)
            return jsonify({"reply": reply})
        except Exception as e:
            logging.exception("Fallback _model.generate(prompt=...) failed: %s", e)
            last_exc = e

    # 2) try top-level genai.generate_text or genai.generate (different client versions)
    if genai is not None:
        for fn_name in ("generate_text", "generate", "text_generation"):
            fn = getattr(genai, fn_name, None)
            if callable(fn):
                try:
                    # many genai helpers accept model=... and prompt=...
                    resp = fn(model="gemini-1.5-flash", prompt=prompt)
                    # try multiple possible response shapes
                    reply = getattr(resp, "text", None) or getattr(resp, "result", None) or str(resp)
                    return jsonify({"reply": reply})
                except Exception as e:
                    logging.exception("genai.%s(...) failed: %s", fn_name, e)
                    last_exc = e

    # 3) try calling generate_content with a single positional draft (some versions accept a different shape)
    if _model is not None:
        try:
            # Some older/newer clients accept a simple input string or dict
            resp = _model.generate_content({"text": prompt})
            reply = getattr(resp, "text", None) or str(resp)
            return jsonify({"reply": reply})
        except Exception as e:
            logging.exception("_model.generate_content({'text': prompt}) failed: %s", e)
            last_exc = e

    # If we reach here, nothing worked — return informative error (but not your key)
    err_msg = "Could not call Gemini with available client methods. See server logs for details."
    if last_exc is not None:
        err_msg += f" Last exception: {type(last_exc).__name__}: {str(last_exc)}"
    return jsonify({"error": err_msg}), 500


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
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
