import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- Load Gemini API Key ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Load latest supported model ---
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        # Gemini expects list of "contents"
        contents = [
            {"role": "user", "parts": [{"text": user_message}]}
        ]

        response = model.generate_content(contents)

        reply = response.text if hasattr(response, "text") else "No response"

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Debug endpoint (optional)
@app.route("/_ls")
def debug_info():
    return jsonify({
        "genai_version": getattr(genai, "__version__", "unknown"),
        "available_models": getattr(genai, "list_models", lambda: "no access")(),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
    
