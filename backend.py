# ==========================================
# backend.py — Serve Sepsis Model via API
# ==========================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ==========================================
# Load the trained model
# ==========================================
try:
    model = joblib.load("sepsis_model.pkl")
    print("✅ Model loaded successfully: sepsis_model.pkl")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ==========================================
# Prediction Route
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract and prepare features (same order as training)
        features = np.array([[
            float(data["heart_rate"]),
            float(data["respiratory_rate"]),
            float(data["temperature"]),
            float(data["mean_arterial_pressure"]),
            float(data["oxygen_saturation"]),
            float(data["white_blood_cell_count"]),
            float(data["lactate_level"]),
            float(data["age"]),
            1 if str(data["sex"]).lower() == "male" else 0
        ]])

        # Get probability of sepsis (class 1)
        prob = model.predict_proba(features)[0][1]
        risk_level = "High Risk" if prob >= 0.35 else "Low Risk"

        # Respond with JSON
        return jsonify({
            "risk_level": risk_level,
            "risk_score": round(float(prob) * 100, 2)  # convert to percentage
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==========================================
# Run the Flask app
# ==========================================
if __name__ == "__main__":
    print("🚀 Flask backend running on http://127.0.0.1:5000")
    app.run(debug=True)

