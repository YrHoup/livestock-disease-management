from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Attempt to load the XGBoost model
model = xgb.XGBClassifier()
try:
    model.load_model('model/animal_disease_xgb_model.json')
    print("Loaded JSON model successfully.")
except Exception as e:
    print(f"Failed to load JSON model: {e}")
    try:
        model.load_model('model/animal_disease_xgb_model.bin')
        print("Loaded binary model successfully.")
    except Exception as e:
        raise Exception(f"Failed to load both JSON and binary models: {e}")

# Define symptom-to-disease mapping for symptom checker
symptom_disease_map = {
    'Fever': ['Foot and Mouth Disease', 'Parvovirus'],
    'Cough': ['Upper Respiratory Infection', 'Kennel Cough'],
    'Lethargy': ['Parvovirus', 'Gastroenteritis', 'Fungal Infection'],
    'Diarrhea': ['Gastroenteritis', 'Parvovirus'],
    'Vomiting': ['Gastroenteritis', 'Parvovirus'],
    'Loss of appetite': ['Parvovirus', 'Fungal Infection'],
    'Difficulty breathing': ['Upper Respiratory Infection'],
    'Lameness': ['Foot and Mouth Disease'],
    'Swelling': ['Fungal Infection'],
    'Skin rash': ['Fungal Infection']
}

# Simple advice for Dangerous prediction
disease_advice = {
    1: "The animal may be dangerous due to a serious condition. Consult a veterinarian immediately.",
    0: "The animal does not appear to be dangerous, but monitor its symptoms and consult a vet if conditions worsen."
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
            # Get form data matching dataset features
            data = {
                'AnimalName': request.form['animal_name'],
                'symptoms1': request.form['symptom_1'],
                'symptoms2': request.form['symptom_2'],
                'symptoms3': request.form['symptom_3'],
                'symptoms4': request.form['symptom_4'],
                'symptoms5': request.form['symptom_5']
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([data])

            # Encode categorical variables (consistent with training)
            categorical_cols = ['AnimalName', 'symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']
            for col in categorical_cols:
                le = LabelEncoder()
                # Fit with known categories (should match training data)
                if col == 'AnimalName':
                    le.fit(['Dog', 'Cat', 'Cow', 'Goat', 'Sheep', 'Pig', 'Horse', 'Rabbit', 'Chicken', 'Duck'])
                else:
                    le.fit(['Fever', 'Cough', 'Lethargy', 'Diarrhea', 'Vomiting', 'Loss of appetite',
                            'Difficulty breathing', 'Lameness', 'Swelling', 'Skin rash'])
                input_df[col] = le.transform(input_df[col])

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][prediction]
            advice = disease_advice.get(prediction, "No advice available.")

            return render_template('predict.html',
                                 prediction='Yes' if prediction == 1 else 'No',
                                 probability=f"{probability * 100:.2f}%",
                                 advice=advice)
        except Exception as e:
            return render_template('predict.html', error=f"Error processing input: {str(e)}")
    return render_template('predict.html')

@app.route('/symptom_checker', methods=['GET', 'POST'])
def symptom_checker():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        possible_diseases = set()
        for symptom in symptoms:
            if symptom in symptom_disease_map:
                possible_diseases.update(symptom_disease_map[symptom])
        return render_template('symptom_checker.html', possible_diseases=list(possible_diseases))
    return render_template('symptom_checker.html')

@app.route('/advice')
def advice():
    return render_template('advice.html', disease_advice=disease_advice)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)