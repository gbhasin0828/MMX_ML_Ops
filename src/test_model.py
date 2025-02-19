import mlflow.pyfunc
import pandas as pd
import os

# ✅ Set MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/gbhasin0828/MMX_MLFlow.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Load the Latest Registered Model
model = mlflow.pyfunc.load_model("models:/Best_Marketing_Model/latest")
print("🚀 Successfully Loaded Model from MLflow")

# ✅ Define the Path for `data_copy.csv`
data_copy_path = "data/data_copy.csv"

# ✅ Check if `data_copy.csv` Exists
if not os.path.exists(data_copy_path):
    raise FileNotFoundError(f"❌ data_copy.csv not found at {data_copy_path}")

# ✅ Load the Training Dataset
df_test = pd.read_csv(data_copy_path)
print(f"📊 Loaded {len(df_test)} rows from data_copy.csv")

# ✅ Select the Same Features Used in Training
feature_columns = ['mdsp_dm', 'mdsp_inst', 'mdsp_nsp', 'mdsp_auddig',
                   'mdsp_audtr', 'mdsp_vidtr', 'mdsp_viddig', 'mdsp_so',
                   'mdsp_on', 'mdsp_sem']

df_features = df_test[feature_columns]

# ✅ Run Model Predictions
predictions = model.predict(df_features)
df_test["predictions"] = predictions

# ✅ Save Predictions as Output
output_path = "data/data_copy_with_predictions.csv"
df_test.to_csv(output_path, index=False)
print(f"🚀 Model Predictions Saved to {output_path}")

# ✅ Print Sample Predictions in GitHub Actions Logs
print(df_test.head(10))  # ✅ Print first 10 rows
