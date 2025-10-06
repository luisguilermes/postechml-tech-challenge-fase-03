import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import shap
import os

# 1. Load from csv or DB
df = pd.read_csv("../data/fc26_players.csv")

# 2. Basic cleaning
df = df.dropna(subset=["value_eur"])  # target required
df['value_eur'] = df['value_eur'].astype(float)

# 3. Feature engineering (exemplo)
numeric_cols = ["age", "overall", "potential", "wage_eur"]
cat_cols = ["club_name", "club_position", "nationality_name"]

# 4. Preprocessing pipeline
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="unknown")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([("num", num_pipeline, numeric_cols), ("cat", cat_pipeline, cat_cols)])

X = df[numeric_cols + cat_cols]
y = df["value_eur"]

# 5. Model + CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes, rmses, r2s = [], [], []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train_t, y_train)
    preds = model.predict(X_val_t)
    maes.append(mean_absolute_error(y_val, preds))
    rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    r2s.append(r2_score(y_val, preds))

print("MAE", np.mean(maes), "RMSE", np.mean(rmses), "R2", np.mean(r2s))

# 6. Refit on full data and save pipeline
os.makedirs("models", exist_ok=True)
full_pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42))])
full_pipeline.fit(X, y)
joblib.dump(full_pipeline, "models/fc26_value_model.joblib")

# 7. SHAP explanation example (opcional)
explainer = shap.Explainer(full_pipeline.named_steps['model'])
# To compute shap we need numeric matrix after preprocessor:
X_t = full_pipeline.named_steps['preprocessor'].transform(X)
shap_values = explainer(X_t[:200])  # amostra rápida
shap.summary_plot(shap_values, features=X_t[:200])