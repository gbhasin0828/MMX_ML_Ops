from flask import Flask, request, jsonify, render_template
import mlflow.pyfunc
import pandas as pd
import os

app = Flask(__name__)

# ✅ Set MLflow Tracking URI for DagsHub
MLFLOW_TRACKING_URI = "https://dagshub.com/gbhasin0828/MMX_MLFlow.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Load the MLflow Model from DagsHub
print("🚀 Loading MLflow Model from DagsHub...")
try:
    model = mlflow.pyfunc.load_model("models:/Best_Marketing_Model_New/latest")
    print("✅ Successfully Loaded Model from MLflow")
except Exception as e:
    print(f"❌ Error Loading Model: {str(e)}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)

    # ✅ Run Model Prediction
    predictions = model.predict(df)
    df["predictions"] = predictions

    # ✅ Convert to JSON Response
    return df.to_json(orient="records"), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
