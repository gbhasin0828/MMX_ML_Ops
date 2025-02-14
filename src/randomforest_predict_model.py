
"""# **Prediction Code**"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import adstock_transform , hill_transform, find_optimal_alpha_per_channel

import os
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_bundle.pkl")
loaded_params = joblib.load(model_path)

model_path = "model_bundle.pkl"

model_bundle = joblib.load(model_path)

rf_model = model_bundle["rf_model"]
media_scaler = model_bundle["media_scaler"]
y_scaler = model_bundle["y_scaler"]
best_alphas = model_bundle["best_alphas"]
best_ec_values = model_bundle["best_ec_values"]
best_slope_values = model_bundle["best_slope_values"]

import numpy as np
import pandas as pd
import joblib

# ✅ Load the Saved Model and Parameters
model_bundle = joblib.load("model_bundle.pkl")

rf_model = model_bundle["rf_model"]
media_scaler = model_bundle["media_scaler"]
y_scaler = model_bundle["y_scaler"]
best_alphas = model_bundle["best_alphas"]
best_ec_values = model_bundle["best_ec_values"]
best_slope_values = model_bundle["best_slope_values"]

# ✅ Define Adstock Transformation
def adstock_transform(X, alpha):
    """Apply Adstock transformation recursively."""
    X_adstocked = np.zeros_like(X, dtype=float)
    X_adstocked[0] = X[0]
    for t in range(1, len(X)):
        X_adstocked[t] = X[t] + alpha * X_adstocked[t - 1]
    return X_adstocked

# ✅ Define Hill Transformation
def hill_transform(X, ec, slope):
    """Hill function transformation."""
    return (X ** slope) / (ec ** slope + X ** slope)

# ✅ Prediction Function
def predict_sales(csv_file):
    """Predict sales using the trained Random Forest model."""

    # Load new data
    df = pd.read_csv(csv_file)

    media_cols = ['mdsp_dm', 'mdsp_inst', 'mdsp_nsp', 'mdsp_auddig', 'mdsp_audtr',
                  'mdsp_vidtr', 'mdsp_viddig', 'mdsp_so', 'mdsp_on', 'mdsp_sem']

    # Check if required columns are present
    missing_cols = [col for col in media_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ✅ Step 1: Apply Scaling (Convert DataFrame to NumPy before transforming)
    df[media_cols] = media_scaler.transform(df[media_cols].values)

    # ✅ Step 2: Extract Media Spend
    X_media = df[media_cols].values

    # ✅ Step 3: Apply Adstock Transformation
    X_media_adstocked = np.zeros_like(X_media, dtype=float)
    for i, alpha in enumerate(best_alphas):
        X_media_adstocked[:, i] = adstock_transform(X_media[:, i], alpha)

    # ✅ Step 4: Apply Hill Transformation
    X_media_hill = np.zeros_like(X_media_adstocked, dtype=float)
    for i in range(len(media_cols)):
        X_media_hill[:, i] = hill_transform(X_media_adstocked[:, i], best_ec_values[i], best_slope_values[i])

    # ✅ Step 5: Predict Sales Using the Random Forest Model
    predicted_sales_scaled = rf_model.predict(X_media_hill)

    # ✅ Step 6: Inverse Transform Predictions to Original Sales Scale
    predicted_sales = y_scaler.inverse_transform(predicted_sales_scaled.reshape(-1, 1)).flatten()

    # ✅ Step 7: Save Predictions to CSV
    df['predicted_sales'] = predicted_sales.astype(float)
    output_file = 'predicted_sales_output.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Predicted sales saved to {output_file}")

    return df[['predicted_sales']]

# ✅ Run Prediction on Example File
file = 'data_copy.csv'
predict_sales(file)
