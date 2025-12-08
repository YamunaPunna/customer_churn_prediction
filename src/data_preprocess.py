import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def build_preprocessor(df, categorical_cols, numeric_cols):
    # numeric pipeline
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # categorical pipeline
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ])
    return preprocessor

def preprocess_and_save(df, target_col='Churn', test_size=0.2, random_state=42):
    # Target
    y = (df[target_col].map({'Yes':1, 'No':0}) 
         if df[target_col].dtype == 'object' else df[target_col])
    X = df.drop(columns=[target_col])

    # Columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessor
    preprocessor = build_preprocessor(df, categorical_cols, numeric_cols)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit preprocessor
    preprocessor.fit(X_train)

    # Save preprocessor + column info
    joblib.dump({
        'preprocessor': preprocessor,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'all_columns': categorical_cols + numeric_cols
    }, '../models/preprocess.joblib')
    print("✔ Preprocess saved successfully")

    # Transform
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    return X_train_t, X_test_t, y_train.values, y_test.values
