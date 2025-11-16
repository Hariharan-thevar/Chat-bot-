# app.py
import os
import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Optional .env for local testing
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try import google generative client
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro")  # default to gemini-pro
_model = None

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Configured genai with API key.")
        try:
            _model = getattr(genai, "GenerativeModel", lambda *a, **k: None)(MODEL_NAME)
            if _model:
                logging.info("Created GenerativeModel handle for: %s", MODEL_NAME)
            else:
                logging.info("GenerativeModel handle unavailable (client may not expose it).")
        except Exception:
            logging.exception("Failed to create GenerativeModel handle (continuing with genai-level fallbacks).")
            _model = None
    except Exception:
        logging.exception("Failed to configure genai (check GEMINI_API_KEY).")
else:
    if GEMINI_API_KEY and genai is None:
        logging.warning("GEMINI_API_KEY set but google.generativeai import failed.")
    else:
        logging.warning("GEMINI_API_KEY not set; /chat will return an error until you add it.")


SYSTEM_PROMPT = "You are a helpful assistant."


def extract_text_from_response(resp: Any) -> str:
    if resp is None:
        return ""
    for attr in ("text", "result", "output_text", "content", "response", "output"):
        try:
            val = getattr(resp, attr, None)
            if val:
                return str(val)
        except Exception:
            pass
    try:
        if isinstance(resp, dict):
            for k in ("text", "result", "output", "content"):
                if k in resp and resp[k]:
                    return str(resp[k])
    except Exception:
        pass
    return str(resp)


def build_prompt_from_messages(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "user").upper()
        if "parts" in m and isinstance(m["parts"], list) and m["parts"]:
            text = m["parts"][0].get("text", "")
        else:
            text = m.get("content", "")
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as exc:
        logging.exception("render_template failed: %s", exc)
        try:
            abs_path = Path(app.root_path) / "templates" / "index.html"
            logging.info("Attempting static fallback: %s", str(abs_path))
            if abs_path.exists():
                return send_file(str(abs_path))
            return ("index.html not found in container", 500)
        except Exception as e2:
            logging.exception("Static fallback failed: %s", e2)
            return ("Template error", 500)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "missing 'message'"}), 400
    if GEMINI_API_KEY is None:
        return jsonify({"error": "GEMINI_API_KEY not configured"}), 500

    # Build messages structure
    messages = [{"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}]
    for h in history:
        if "content" in h:
            messages.append({"role": h.get("role", "user"), "parts": [{"text": h.get("content", "")}]})
        else:
            messages.append({"role": h.get("role", "user"), "parts": h.get("parts", [{"text": ""}])})
    messages.append({"role": "user", "parts": [{"text": message}]})

    prompt = build_prompt_from_messages(messages)
    last_exc = None

    # ----- First attempt: call _model.generate_content with 'contents' (log showed 'Did you mean contents' hint) -----
    if _model is not None:
        try:
            # Try several 'contents' shapes commonly accepted:
            attempts = [
                {"contents": [prompt]},                      # simple list of strings
                {"contents": [{"text": prompt}]},            # list of dicts with text
                {"contents": [{"type": "text", "text": prompt}]},  # richer dict
                {"contents": [{"input": prompt}]},           # alternate key name
                {"contents": [{"content": prompt}]},         # alternate key name
            ]
            for args in attempts:
                try:
                    logging.info("Trying _model.generate_content with args: %s", args)
                    resp = _model.generate_content(**args)
                    reply = extract_text_from_response(resp)
                    return jsonify({"reply": reply})
                except TypeError as te:
                    logging.info("Signature mismatch for args %s: %s", args, te)
                    last_exc = te
                except Exception as e:
                    logging.info("Attempt with args %s failed: %s", args, e)
                    last_exc = e
        except Exception as e:
            logging.exception("Error in generate_content 'contents' attempts: %s", e)
            last_exc = e

    # ----- Second attempt: try _model.generate or other model-level methods with the prompt -----
    if _model is not None:
        try:
            if hasattr(_model, "generate"):
                try:
                    logging.info("Trying _model.generate(prompt=...)")
                    resp = _model.generate(prompt=prompt)
                    reply = extract_text_from_response(resp)
                    return jsonify({"reply": reply})
                except Exception as e:
                    logging.info("_model.generate failed: %s", e)
                    last_exc = e

            for method_name in ("generate_text", "generateText", "text_generation", "create_text"):
                mfn = getattr(_model, method_name, None)
                if callable(mfn):
                    try:
                        logging.info("Trying _model.%s(prompt=...)", method_name)
                        resp = mfn(prompt=prompt)
                        reply = extract_text_from_response(resp)
                        return jsonify({"reply": reply})
                    except Exception as e:
                        logging.info("%s failed: %s", method_name, e)
                        last_exc = e
        except Exception as e:
            logging.exception("Model-level calls failed: %s", e)
            last_exc = e

    # ----- Third attempt: genai-level helpers (various names and arg shapes) -----
    if genai is not None:
        genai_fn_names = ["generate_text", "generate", "text_generation", "generateText", "text"]
        for name in genai_fn_names:
            fn = getattr(genai, name, None)
            if callable(fn):
                arg_variants = [
                    {"model": MODEL_NAME, "prompt": prompt},
                    {"prompt": prompt, "model": MODEL_NAME},
                    {"prompt": prompt},
                    {"text": prompt},
                    {"input": prompt},
                    {"prompts": [prompt], "model": MODEL_NAME},
                ]
                for args in arg_variants:
                    try:
                        logging.info("Trying genai.%s with args: %s", name, args)
                        resp = fn(**args)
                        reply = extract_text_from_response(resp)
                        return jsonify({"reply": reply})
                    except Exception as e:
                        logging.info("genai.%s with args %s failed: %s", name, args, e)
                        last_exc = e

    # ----- Final fallback: try a simple positional call to generate_content if available -----
    if _model is not None:
        try:
            try:
                logging.info("Trying positional _model.generate_content with single arg (prompt)")
                resp = _model.generate_content(prompt)
                reply = extract_text_from_response(resp)
                return jsonify({"reply": reply})
            except Exception as e:
                logging.info("Positional generate_content failed: %s", e)
                last_exc = e
        except Exception as e:
            logging.exception("Final positional generate_content attempt raised: %s", e)
            last_exc = e

    # Nothing worked
    err_msg = "Could not call Gemini with available client methods. Check server logs."
    if last_exc is not None:
        err_msg = f"{err_msg} Last exception: {type(last_exc).__name__}: {str(last_exc)}"
    logging.error(err_msg)
    return jsonify({"error": err_msg}), 500


@app.route("/_ls")
def _ls():
    root = Path.cwd()
    files = [p.name for p in root.iterdir() if p.is_file()]
    dirs = [p.name for p in root.iterdir() if p.is_dir()]
    tdir = root / "templates"
    tfiles = [p.name for p in tdir.iterdir()] if tdir.exists() else []
    genai_attrs = [] if genai is None else sorted([a for a in dir(genai) if not a.startswith("_")])[:200]
    return jsonify({
        "cwd": str(root),
        "root_files": files,
        "root_dirs": dirs,
        "templates_files": tfiles,
        "genai_attrs": genai_attrs
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
        
