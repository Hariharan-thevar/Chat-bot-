# app.py
import os
import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# Optional: load .env locally for development
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Import google generative client (may be missing/different versions)
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro")  # default to gemini-pro (older runtimes)
_model = None

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Configured genai with API key (not shown).")
        # Try to create a model handle if available
        try:
            # Many versions expose GenerativeModel
            _model = getattr(genai, "GenerativeModel", lambda *a, **k: None)(MODEL_NAME)
            if _model is None:
                # fallback: some libraries accept a factory
                logging.info("GenerativeModel returned None or not supported in this client.")
            else:
                logging.info("Created model handle for: %s", MODEL_NAME)
        except Exception:
            logging.exception("Could not create a GenerativeModel handle (this is OK).")
            _model = None
    except Exception:
        logging.exception("Failed to configure genai (check GEMINI_API_KEY and client).")
else:
    if GEMINI_API_KEY and genai is None:
        logging.warning("GEMINI_API_KEY present but google.generativeai is not importable.")
    else:
        logging.warning("GEMINI_API_KEY not set; /chat will return an informative error.")


SYSTEM_PROMPT = "You are a helpful assistant."


def _try_calls(fn: Callable[..., Any], arg_variants: List[dict]) -> Tuple[bool, Any, Exception]:
    """
    Try calling fn with each variant dict in arg_variants.
    Returns (success, response, last_exception)
    """
    last_exc: Exception = None  # type: ignore
    for args in arg_variants:
        try:
            logging.info("Trying %s with args %s", getattr(fn, "__name__", str(fn)), args)
            resp = fn(**args) if isinstance(args, dict) else fn(args)
            return True, resp, None
        except Exception as e:
            last_exc = e
            logging.info("Attempt failed: %s", e)
    return False, None, last_exc


def extract_text_from_response(resp: Any) -> str:
    """
    Try to extract a human-readable text reply from various response shapes.
    """
    # Common shapes: resp.text, resp.result, resp.output_text, resp[0], str(resp)
    if resp is None:
        return ""
    for attr in ("text", "result", "output_text", "content", "response"):
        try:
            val = getattr(resp, attr, None)
            if val:
                return str(val)
        except Exception:
            pass
    try:
        # some responses are dict-like
        if isinstance(resp, dict):
            for k in ("text", "result", "output", "content"):
                if k in resp and resp[k]:
                    return str(resp[k])
    except Exception:
        pass
    # fallback to string representation
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
            logging.info("Attempting static fallback path: %s", str(abs_path))
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
        return jsonify({"error": "GEMINI_API_KEY not configured in environment."}), 500

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
    # -------------- Try genai-level helper functions (various names & arg shapes) --------------
    if genai is not None:
        genai_fn_names = ["generate_text", "generate", "text_generation", "text", "generateText"]
        for name in genai_fn_names:
            fn = getattr(genai, name, None)
            if callable(fn):
                # try several arg shapes
                arg_variants = [
                    {"model": MODEL_NAME, "prompt": prompt},
                    {"prompt": prompt, "model": MODEL_NAME},
                    {"prompt": prompt},
                    {"text": prompt},
                    {"input": prompt},
                    {"prompts": [prompt], "model": MODEL_NAME},
                ]
                success, resp, exc = _try_calls(fn, arg_variants)
                if success:
                    reply = extract_text_from_response(resp)
                    return jsonify({"reply": reply})
                last_exc = exc

    # -------------- Try model-level calls on _model (if available) --------------
    if _model is not None:
        # Try several ways to call generate_content / generate / generate_text
        # 1) try generate_content with different shapes
        try:
            gen_fn = getattr(_model, "generate_content", None)
            if callable(gen_fn):
                arg_variants = [
                    ({"messages": messages}),
                    ({"input": messages}),
                    ({"text": prompt}),
                    ({"prompt": prompt}),
                    ({"content": prompt}),
                    ({"instances": [{"input": prompt}]})
                ]
                last_exc_here = None
                for args in arg_variants:
                    try:
                        logging.info("Trying _model.generate_content with args: %s", args)
                        resp = gen_fn(**args)
                        reply = extract_text_from_response(resp)
                        return jsonify({"reply": reply})
                    except Exception as e:
                        last_exc_here = e
                        logging.info("generate_content attempt failed: %s", e)
                last_exc = last_exc_here
        except Exception as e:
            logging.exception("Error while trying _model.generate_content: %s", e)
            last_exc = e

        # 2) try _model.generate(prompt=...)
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
            # 3) try other model-level method names
            for method_name in ("generate_text", "generateText", "text_generation"):
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
            logging.exception("Model-level attempts raised: %s", e)
            last_exc = e

    # -------------- Last fallback: try a simple HTTP-style usage via genai if any client has a 'client' attr --------------
    # e.g., some clients expose genai.Client() or genai.client
    try:
        client_candidates = []
        if genai is not None:
            for candidate_name in ("Client", "client", "GenAIClient"):
                c = getattr(genai, candidate_name, None)
                if c:
                    client_candidates.append(c)
        for cc in client_candidates:
            try:
                # try instantiating if callable
                client = cc() if callable(cc) else cc
                # try common method names
                for method in ("generate_text", "generate", "text_generation"):
                    m = getattr(client, method, None)
                    if callable(m):
                        try:
                            logging.info("Trying client.%s(prompt=...)", method)
                            resp = m(model=MODEL_NAME, prompt=prompt)
                            reply = extract_text_from_response(resp)
                            return jsonify({"reply": reply})
                        except Exception as e:
                            logging.info("client.%s attempt failed: %s", method, e)
                            last_exc = e
            except Exception as e:
                logging.info("Client candidate instantiation failed: %s", e)
                last_exc = e
    except Exception:
        pass

    # Nothing worked
    err_msg = "Could not call Gemini with available client methods. Check server logs for details."
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
    info = {
        "cwd": str(root),
        "root_files": files,
        "root_dirs": dirs,
        "templates_files": tfiles,
        "genai_attrs": [] if genai is None else sorted([a for a in dir(genai) if not a.startswith("_")])[:150],
    }
    return jsonify(info)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
