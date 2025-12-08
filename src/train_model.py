import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from data_preprocess import load_data, preprocess_and_save

def train(path_to_csv='data/telecom_churn.csv'):
    df = load_data(path_to_csv)
    X_train, X_test, y_train, y_test = preprocess_and_save(df)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    # predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(model, '../models/xgb_model.joblib')
    print("✔ Model saved to models/xgb_model.joblib")

if __name__ == "__main__":
    train()
