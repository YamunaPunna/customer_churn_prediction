from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from predict_util import predict_single

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '../app/templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '../app/static')
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()

        # Convert numeric fields
        numeric_fields = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
        for field in numeric_fields:
            if field in form_data:
                form_data[field] = float(form_data[field])

        # Predict
        result_dict = predict_single(form_data)
        result = "Churn" if result_dict['prediction'] == 1 else "No Churn"
        probability = round(result_dict['probability']*100, 2)

        return render_template('result.html', result=result, prob=probability)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
