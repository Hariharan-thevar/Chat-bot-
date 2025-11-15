from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import google.generativeai as genai

app = Flask(__name__, template_folder="templates")
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Put it in your environment or in a .env file (do NOT commit your real key).")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = "You are a helpful assistant."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "missing 'message' field"}), 400

    user_msg = data["message"]
    history = data.get("history", [])

    messages = [
        {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]}
    ]
    for h in history:
        messages.append({"role": h.get("role", "user"), "parts": [{"text": h.get("content", "")}]})

    messages.append({"role": "user", "parts": [{"text": user_msg}]})

    try:
        response = model.generate_content(messages=messages)
        reply = getattr(response, "text", None)
        if reply is None:
            reply = str(response)

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
