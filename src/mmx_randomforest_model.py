
import sklearn
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils import adstock_transform , hill_transform, find_optimal_alpha_per_channel
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data_copy.csv")
media_cols = ['mdsp_dm', 'mdsp_inst', 'mdsp_nsp', 'mdsp_auddig', 'mdsp_audtr',
              'mdsp_vidtr', 'mdsp_viddig', 'mdsp_so', 'mdsp_on', 'mdsp_sem']
y_sales = df['sales'].values.reshape(-1, 1)

# ✅ Scale media spend
media_scaler = MinMaxScaler()
X_media = media_scaler.fit_transform(df[media_cols].values)

# ✅ Define Parameter Ranges
alpha_range = np.arange(0.2, 0.9, 0.1)  # 🚀 Optimize alpha between 0.2 and 0.8
ec_range = np.linspace(0.1, 1.0, 10)   # 🚀 Optimize ec between 10% and 100%
slope_range = np.tan(np.radians(np.arange(25, 80, 10)))  # 🚀 tan(θ) between 25° and 75°

# ✅ Initialize Best Parameters
best_alphas = np.zeros(len(media_cols))
best_ec_values = np.zeros(len(media_cols))
best_slope_values = np.zeros(len(media_cols))

# ✅ Step 1: Optimize `alpha` (Adstock Decay) Per Media Channel
best_alphas = np.zeros(len(media_cols))
X_media_adstocked = np.zeros_like(X_media)

for i in range(len(media_cols)):
    best_r2 = -np.inf
    best_alpha = None

    for alpha in alpha_range:
        X_temp_adstocked = adstock_transform(X_media[:, i], alpha).reshape(-1, 1)

        # ✅ Fit Linear Regression to Evaluate R²
        model = LinearRegression()
        model.fit(X_temp_adstocked, y_sales)
        y_pred = model.predict(X_temp_adstocked)
        r2 = r2_score(y_sales, y_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    best_alphas[i] = best_alpha
    X_media_adstocked[:, i] = adstock_transform(X_media[:, i], best_alpha)
    print(f"✅ Best Alpha for {media_cols[i]}: {best_alpha}, Best R²: {best_r2}")

for i in range(len(media_cols)):
    best_r2 = -np.inf
    best_ec, best_slope = None, None

    for ec in ec_range:
        for slope in slope_range:
            X_transformed = hill_transform(X_media_adstocked[:, i], ec, slope).reshape(-1, 1)

            # ✅ Fit Linear Regression to Evaluate R²
            model = LinearRegression()
            model.fit(X_transformed, y_sales)
            y_pred = model.predict(X_transformed)
            r2 = r2_score(y_sales, y_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_ec, best_slope = ec, slope

    best_ec_values[i] = best_ec
    best_slope_values[i] = best_slope
    print(f"✅ Best ec for {media_cols[i]}: {best_ec}, Best slope: {best_slope}, Best R²: {best_r2}")

X_media_hill = np.zeros_like(X_media_adstocked)

for i in range(len(media_cols)):
    X_media_hill[:, i] = hill_transform(X_media_adstocked[:, i], best_ec_values[i], best_slope_values[i])

y_scaler = MinMaxScaler()
y_sales_scaled = y_scaler.fit_transform(y_sales)

from sklearn.ensemble import RandomForestRegressor

# ✅ Step 1: Initialize Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=500,  
    max_depth=5,       # Keep similar depth as before
    random_state=42
)

# ✅ Step 2: Train Random Forest Model on Transformed Data
rf_model.fit(X_media_hill, y_sales_scaled)

# ✅ Step 3: Save Random Forest Model & Parameters
joblib.dump({
    "rf_model": rf_model,  # ✅ Save Random Forest model instead of XGBoost
    "media_scaler": media_scaler,
    "y_scaler": y_scaler,
    "best_alphas": best_alphas,
    "best_ec_values": best_ec_values,
    "best_slope_values": best_slope_values},
"model_bundle.pkl")

print("✅ Random Forest model and optimized parameters saved successfully!")

from sklearn.metrics import r2_score

# ✅ Step 1: Predict on Training Data
y_pred_scaled = rf_model.predict(X_media_hill)

# ✅ Step 2: Inverse Transform Predictions to Actual Sales
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# ✅ Step 3: Compute `R²` Score
final_r2 = r2_score(y_sales, y_pred)
print(f"✅ Final Model R² on Training Data: {final_r2:.4f}")

