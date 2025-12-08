import joblib
import pandas as pd

# Load model and preprocessing
model = joblib.load("../models/xgb_model.joblib")
preprocess_data = joblib.load("../models/preprocess.joblib")
preprocessor = preprocess_data["preprocessor"]
all_columns = preprocess_data["all_columns"]
numeric_cols = preprocess_data["numeric_cols"]

def predict_single(input_dict):
    df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in all_columns:
        if col not in df.columns:
            df[col] = 0 if col in numeric_cols else 'No'

    # Ensure correct column order
    df = df[all_columns]

    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]
    pred = int(prob > 0.5)

    return {"prediction": pred, "probability": float(prob)}
