from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Load the XGBoost model
model = xgb.XGBClassifier()
try:
    model.load_model('animal_disease_xgb_model.json')
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise Exception(f"Failed to load model: {e}")

# Define advice for prediction
disease_advice = {
    1: "The animal may have a serious condition. Isolate it and consult a veterinarian immediately.",
    0: "The animal appears healthy, but continue monitoring for any changes in behavior or symptoms."
}


@app.route('/')
def home():
    # Random health tip
    tips = [
        "Ensure animals have clean water to prevent disease spread.",
        "Provide a balanced diet to boost immunity.",
        "Keep living areas clean to reduce infection risk."
    ]
    health_tip = np.random.choice(tips)
    return render_template('home.html', health_tip=health_tip)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data matching model features
            data = {
                'AnimalName': request.form['AnimalName'],
                'symptoms1': request.form['symptoms1'],
                'symptoms2': request.form['symptoms2'],
                'symptoms3': request.form['symptoms3'],
                'symptoms4': request.form['symptoms4'],
                'symptoms5': request.form['symptoms5']
            }

            # Convert to DataFrame with correct feature order
            input_df = pd.DataFrame([data], columns=['AnimalName', 'symptoms1', 'symptoms2',
                                                     'symptoms3', 'symptoms4', 'symptoms5'])

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] * 100  # Probability of being dangerous

            # Generate more specific advice based on symptoms
            symptoms = [data[f'symptoms{i}'] for i in range(1, 6) if data[f'symptoms{i}'] != 'No']
            advice = generate_advice(prediction, data['AnimalName'], symptoms)

            return render_template('predict.html',
                                   prediction='Yes' if prediction == 1 else 'No',
                                   probability=f"{probability:.2f}",
                                   advice=advice)
        except Exception as e:
            return render_template('predict.html', error=f"Error processing input: {str(e)}")
    return render_template('predict.html')


def generate_advice(prediction, animal_type, symptoms):
    if prediction == 1:  # Dangerous
        symptom_list = ", ".join(symptoms)
        return f"Your {animal_type} shows concerning symptoms ({symptom_list}). Isolate the animal immediately and contact a veterinarian."
    else:
        return f"Your {animal_type} appears healthy based on the reported symptoms. Continue regular monitoring."


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)