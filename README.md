'''*Customer Churn Prediction Using XGBoost*
This project is a machine-learning-based web application that predicts whether a telecom customer will churn (leave the service).
The prediction is made using an XGBoost model, and a Flask web interface is used to interact with the model.

*Features*:
->Real-time churn prediction
->Probability-based prediction output
->Clean Flask web UI
->Preprocessing included
->Well-structured project
->Model trained using XGBoost

*Project Structure*
customer_churn/
│── src/
│   ├── app.py
│   └── predict_util.py
│
│── models/
│   ├── xgb_model.joblib
│   └── preprocess.joblib
│
│── templates/
│   ├── index.html
│   └── result.html
│
│── static/
│   └── style.css
│
│── data/
│   └── telecom_churn_prediction.csv
│
│── README.md

*How to Run the Project*:
1.Activate Virtual Environment
    venv\Scripts\activate

2️.Install Requirements
  pip install -r requirements.txt

3️.Train Model (if needed)
python train.py

4️.Run Flask App
python src/app.py


Then open:
  http://127.0.0.1:5000/


*Model*:
Algorithm: XGBoost
->Output: Churn / Not Churn + Probability
->Preprocessing: One-hot encoding, scaling

*Deployment Options*:
You can deploy this project on:
->Render
->Railway
->Netlify + Flask API
->AWS EC2

*Technologies Used*:
->Python
->Flask
->XGBoost
->HTML + CSS
->Joblib

*Project Summary*:
 This project predicts telecom customer churn with high accuracy using XGBoost, and provides a simple and interactive web interface built with Flask.'''