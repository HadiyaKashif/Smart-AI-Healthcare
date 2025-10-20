import os
import sys
import joblib
import requests
import traceback
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Allow imports from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Src_Code.rag_integration import query_rag

load_dotenv()
app = Flask(__name__)

# ================== Load ML Model ==================
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/decision_tree_model.pkl"))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# ================== Email Alert ==================
def send_email_alert(to_email, risk, causes, suggestions):
    try:
        sender = os.getenv("EMAIL_SENDER")
        password = os.getenv("EMAIL_PASSWORD")

        # Map risk to color + emoji
        risk_colors = {
            "Good": ("#4CAF50", "🟢"),
            "Fair": ("#FFC107", "🟠"),
            "Bad": ("#F44336", "🔴")
        }
        color, emoji = risk_colors.get(risk, ("#9E9E9E", "⚪"))

        subject = f"{emoji} Smart Health Alert: {risk} Risk Detected"

        # Trim long text if needed
        def shorten_text(text, limit=400):
            return text[:limit] + "..." if len(text) > limit else text

        causes_html = "".join(f"<li>{shorten_text(c)}</li>" for c in causes[:3])
        suggestions_html = "".join(f"<li>{shorten_text(s)}</li>" for s in suggestions[:3])

        # Stylish HTML template
        body = f"""
        <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; margin:0; padding:0; background-color:#f5f7fa;">
            <div style="max-width:600px; margin:30px auto; background:#fff; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.1); overflow:hidden;">
                
                <div style="background:{color}; color:white; text-align:center; padding:16px 20px; font-size:20px; font-weight:bold;">
                    {emoji} Health Risk Level: {risk}
                </div>

                <div style="padding:20px;">
                    <p>Dear User,</p>
                    <p>Our system has detected a <strong>{risk}</strong> health risk based on your recent vitals.</p>

                    <h3 style="color:{color}; margin-top:20px;">🧠 Possible Causes</h3>
                    <ul style="line-height:1.5; color:#333;">
                        {causes_html or "<li>No detailed causes found.</li>"}
                    </ul>

                    <h3 style="color:{color}; margin-top:20px;">💡 Suggestions</h3>
                    <ul style="line-height:1.5; color:#333;">
                        {suggestions_html or "<li>Consult a doctor for personalized advice.</li>"}
                    </ul>

                    <div style="text-align:center; margin-top:30px;">
                        <a href="https://your-app-url.com/health-report" target="_blank" 
                           style="background:{color}; color:white; text-decoration:none; padding:12px 24px; border-radius:6px; font-weight:bold;">
                           View Full Health Report
                        </a>
                    </div>

                    <p style="margin-top:30px; color:#666; font-size:14px;">
                        Stay safe and healthy,<br>
                        — <strong>Smart Health Assistant</strong>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["From"] = sender
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        # SMTP Send
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)

        print(f"📧 Alert email sent to {to_email}")

    except Exception as e:
        print("⚠️ Email send failed:", e)

# ================== OpenStreetMap Doctor Search ==================
def find_nearby_doctors(lat, lon, radius_km=10):
    """Use Overpass API to find nearby doctors within a given radius"""
    try:
        query = f"""
        [out:json];
        (
          node["amenity"="doctors"](around:{radius_km * 1000},{lat},{lon});
          node["healthcare"="doctor"](around:{radius_km * 1000},{lat},{lon});
        );
        out body;
        """
        response = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=15)
        data = response.json()

        doctors = []
        for element in data.get("elements", [])[:5]:  # limit to 5 doctors
            name = element.get("tags", {}).get("name", "Unnamed Doctor")
            specialty = element.get("tags", {}).get("specialty", "General Practitioner")
            email = element.get("tags", {}).get("email", "N/A")
            distance_km = round(radius_km * 0.3 + (0.5 * len(doctors)), 1)  # fake distance approximation

            doctors.append({
                "name": name,
                "specialty": specialty,
                "distance_km": distance_km,
                "email": email
            })
        return doctors

    except Exception as e:
        print("⚠️ Doctor lookup failed:", e)
        return []

# ================== Risk Prediction ==================
def classify_risk(input_data: dict) -> str:
    try:
        # Encode gender numerically
        gender_str = str(input_data.get("Gender", "")).lower()
        gender = 1 if gender_str in ["female", "f"] else 2  # female=1, male=2

        age = float(input_data.get("Age", 0))
        systolic = float(input_data.get("Systolic BP", 0))
        diastolic = float(input_data.get("Diastolic BP", 0))
        cholesterol = float(input_data.get("Cholesterol", 0))
        bmi = float(input_data.get("BMI", 0))
        smoker = int(bool(input_data.get("Smoker", False)))
        diabetes = int(bool(input_data.get("Diabetes", False)))

        features = [gender, age, systolic, diastolic, cholesterol, bmi, smoker, diabetes]

        pred = model.predict([features])[0]
        return pred

    except Exception as e:
        print("Prediction error:", e)
        return "Unknown"


# ================== Main Endpoint ==================
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON provided"}), 400

        # Validate inputs
        required = [
            "Gender", "Age", "Systolic BP", "Diastolic BP",
            "Cholesterol", "BMI", "Smoker", "Diabetes",
            "Email", "Latitude", "Longitude"
        ]
        missing = [r for r in required if r not in data]
        if missing:
            return jsonify({"error": "Missing fields", "missing": missing}), 400

        # Step 1️⃣ Predict Risk
        risk = classify_risk(data)

        # Step 2️⃣ Get Causes & Suggestions from RAG
        rag_result = query_rag(data, risk)
        causes = rag_result.get("causes", [])
        suggestions = rag_result.get("suggestions", [])

        # Step 3️⃣ Nearby Doctors via OpenStreetMap
        doctors = find_nearby_doctors(data["Latitude"], data["Longitude"])

        # Step 4️⃣ Send Alert if Risk is High
        if risk == "Bad":
            send_email_alert(data["Email"], risk, causes, suggestions)
        # Step 5️⃣ Respond
        response = {
            "risk": risk,
            "causes": causes,
            "suggestions": suggestions,
            "doctors": doctors
        }
        return jsonify(response)

    except Exception as e:
        print("❌ Error in /analyze:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "Smart Health API Running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)