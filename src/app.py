from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc
import pandas as pd
import os
import logging

# ✅ Enable Logging for Debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Set MLflow Tracking URI for DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/gbhasin0828/MMX_MLFlow.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Fetch the Latest Model Version Dynamically
client = mlflow.tracking.MlflowClient()
model_versions = client.search_model_versions(f"name='Best_Marketing_Model_New'")

if model_versions:
    latest_version = sorted(model_versions, key=lambda x: int(x.version), reverse=True)[0].version
    model_uri = f"models:/Best_Marketing_Model_New/{latest_version}"
    logger.info(f"✅ Using MLflow Model Version: {latest_version}")
else:
    raise ValueError("❌ No versions found for 'Best_Marketing_Model_New'")

# ✅ Load the MLflow Model
try:
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("✅ Successfully Loaded Model from MLflow")
except Exception as e:
    logger.error(f"❌ Failed to load MLflow model: {str(e)}")
    exit(1)  # ✅ Stop the app if model loading fails

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")

    if file is None:
        return jsonify({"error": "❌ No file uploaded"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"❌ Error reading CSV: {str(e)}"}), 400

    if df.empty:
        return jsonify({"error": "❌ Uploaded CSV is empty"}), 400

    # ✅ Check for missing columns
    required_columns = ['mdsp_dm', 'mdsp_inst', 'mdsp_nsp', 'mdsp_auddig',
                        'mdsp_audtr', 'mdsp_vidtr', 'mdsp_viddig', 'mdsp_so',
                        'mdsp_on', 'mdsp_sem']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return jsonify({"error": f"❌ Missing columns: {missing_columns}"}), 400

    # ✅ Run Model Prediction
    try:
        predictions = model.predict(df)
        df["predictions"] = predictions
    except Exception as e:
        return jsonify({"error": f"❌ Model Prediction Error: {str(e)}"}), 500

    # ✅ Convert to JSON Response
    return df.to_json(orient="records"), 200

# ✅ Expose API Using ngrok (For GitHub Actions Testing)
if os.getenv("USE_NGROK") == "True":
    from pyngrok import ngrok
    public_url = ngrok.connect(5000).public_url
    logger.info(f"🚀 Flask API is live at: {public_url}")
    print(f"🚀 Flask API is live at: {public_url}")  # ✅ Explicitly prints URL in logs



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
