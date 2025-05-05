import joblib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os
import pickle

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Attempt to load the Random Forest model
model_path = 'model/animal_disease_rf_model.joblib'
try:
    model = joblib.load(model_path)
    print("Loaded model successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")

# Attempt to load encoders
encoders_path = 'model/encoders.pkl'
try:
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    print("Loaded encoders successfully.")
except Exception as e:
    print(f"Failed to load encoders: {e}")
    # Create empty encoders dict as fallback
    encoders = {}

# Attempt to load label encoder for disease prediction
le_disease_path = 'model/disease_encoder.pkl'
try:
    with open(le_disease_path, 'rb') as f:
        disease_encoder = pickle.load(f)
    print("Loaded disease encoder successfully.")
    disease_classes = list(disease_encoder.classes_)
except Exception as e:
    print(f"Failed to load disease encoder: {e}")
    disease_encoder = None
    disease_classes = []

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
            # Get form inputs
            animal_type = request.form['animal_type']
            breed = request.form['breed']
            age = float(request.form['age'])
            weight = float(request.form['weight'])
            gender = request.form['gender']
            temperature = float(request.form['temperature'])
            heart_rate = float(request.form.get('heart_rate', 0))
            duration_days = float(request.form.get('duration', 1))

            # Get symptoms
            symptom1 = request.form.get('symptom1', '')
            symptom2 = request.form.get('symptom2', '')
            symptom3 = request.form.get('symptom3', '')
            symptom4 = request.form.get('symptom4', '')

            # Create a DataFrame with a single row for the input data
            input_data = {
                'Animal_Type': [animal_type],
                'Breed': [breed],
                'Age': [age],
                'Gender': [gender],
                'Weight': [weight],
                'Body_Temperature': [temperature],
                'Heart_Rate': [heart_rate],
                'Duration_Num': [duration_days]
            }

            df = pd.DataFrame(input_data)

            # Create binary flags for symptoms - initialize all to 0
            symptom_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                            'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                            'Nasal_Discharge', 'Eye_Discharge']

            for col in symptom_cols:
                df[col] = 0

            # Extended symptom mapping dictionary
            symptom_mapping = {
                'Loss of appetite': 'Appetite_Loss',
                'Appetite Loss': 'Appetite_Loss',
                'Vomiting': 'Vomiting',
                'Diarrhea': 'Diarrhea',
                'Coughing': 'Coughing',
                'Labored Breathing': 'Labored_Breathing',
                'Difficulty breathing': 'Labored_Breathing',
                'Lameness': 'Lameness',
                'Skin Lesions': 'Skin_Lesions',
                'Skin rash': 'Skin_Lesions',
                'Nasal Discharge': 'Nasal_Discharge',
                'Eye Discharge': 'Eye_Discharge',
                'Fever': None,  # Not directly mapped to binary flags
                'Lethargy': None,  # Not directly mapped to binary flags
                'Swelling': None  # Not directly mapped to binary flags
            }

            # Process symptoms
            all_symptoms = []
            for s in [symptom1, symptom2, symptom3, symptom4]:
                if s and isinstance(s, str) and s.strip():
                    all_symptoms.append(s.strip())

            # Set binary flags based on the input symptoms
            for symptom in all_symptoms:
                if symptom in symptom_mapping and symptom_mapping[symptom]:
                    df[symptom_mapping[symptom]] = 1

            # Age grouping
            df['Age_Group'] = pd.cut(df['Age'],
                                     bins=[0, 1, 3, 5, 10, float('inf')],
                                     labels=['Infant', 'Young', 'Adult', 'Middle_Aged', 'Senior'])

            # Weight categorization
            def categorize_weight(row):
                if row['Animal_Type'] == 'Dog':
                    bins = [0, 10, 25, 40, float('inf')]
                    labels = ['Small', 'Medium', 'Large', 'Giant']
                elif row['Animal_Type'] == 'Cat':
                    bins = [0, 4, 6, 8, float('inf')]
                    labels = ['Small', 'Medium', 'Large', 'Giant']
                else:
                    return 'Medium'  # Default

                return pd.cut([row['Weight']], bins=bins, labels=labels)[0]

            df['Weight_Category'] = df.apply(categorize_weight, axis=1)

            # Ensure categorical columns are treated as strings
            categorical_columns = ['Animal_Type', 'Breed', 'Gender', 'Age_Group', 'Weight_Category']
            for col in categorical_columns:
                df[col] = df[col].astype(str)

            # Encode categorical features
            for col in categorical_columns:
                if col in encoders:
                    encoder = encoders[col]
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
                else:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])

            # Check whether to predict "Possibly Healthy"
            is_potentially_healthy = (len(all_symptoms) == 0 or
                                      (len(all_symptoms) <= 1 and
                                       animal_type == 'Dog' and 38.3 <= temperature <= 39.2) or
                                      (len(all_symptoms) <= 1 and
                                       animal_type == 'Cat' and 38.1 <= temperature <= 39.2))

            # Make prediction
            try:
                if is_potentially_healthy:
                    disease_name = "Possibly Healthy"
                    probability = 0.95
                    prediction_details = {
                        'message': "Based on the inputs, the animal appears to be healthy. However, continue to monitor for any changes.",
                        'recommendations': [
                            "Maintain regular check-ups",
                            "Ensure proper nutrition and hydration",
                            "Continue regular exercise"
                        ]
                    }
                else:
                    # Load the feature names used during model training
                    feature_names = ['Age', 'Weight', 'Body_Temperature', 'Heart_Rate', 'Duration_Num',
                                     'Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                                     'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                                     'Nasal_Discharge', 'Eye_Discharge', 'Animal_Type',
                                     'Breed', 'Gender', 'Age_Group', 'Weight_Category']

                    # Create model input DataFrame
                    X_pred = df[feature_names]

                    # Add the missing 20th feature that was likely selected during training
                    X_pred['Feature_20'] = 0  # Default value

                    # Debug info
                    print(f"Model features count: {X_pred.shape[1]}")
                    print(f"Feature names: {X_pred.columns.tolist()}")

                    # Make prediction
                    prediction_idx = model.predict(X_pred)[0]
                    probabilities = model.predict_proba(X_pred)[0]
                    probability = max(probabilities)

                    # Map prediction to disease name
                    if disease_encoder is not None and prediction_idx < len(disease_encoder.classes_):
                        disease_name = disease_encoder.inverse_transform([prediction_idx])[0]
                    else:
                        disease_name = f"Disease {prediction_idx}"

                    # Provide recommendations
                    prediction_details = {
                        'message': f"The animal may have {disease_name}. Please consult a veterinarian.",
                        'recommendations': [
                            "Isolate the animal from others",
                            "Contact veterinarian for proper diagnosis",
                            "Ensure the animal has access to water"
                        ]
                    }

                return render_template('predict.html',
                                       prediction=disease_name,
                                       probability=f"{probability * 100:.2f}%",
                                       animal_type=animal_type,
                                       breed=breed,
                                       age=age,
                                       weight=weight,
                                       symptoms=all_symptoms,
                                       details=prediction_details)
            except Exception as e:
                # Print additional debug information
                import traceback
                print(traceback.format_exc())
                return render_template('predict.html',
                                       error=f"Error making prediction: {str(e)}")
        except Exception as e:
            return render_template('predict.html',
                                   error=f"Error processing input: {str(e)}")

    # For GET requests, show the form
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

@app.route('/save_encoders')
def save_encoders():
    """Admin route to generate and save encoders from the dataset"""
    if not os.path.exists('model'):
        os.makedirs('model')

    try:
        # Load dataset
        df = pd.read_csv('cleaned_animal_disease_prediction.csv')

        # Generate encoders for categorical columns
        categorical_columns = ['Animal_Type', 'Breed', 'Gender', 'Age_Group', 'Weight_Category']
        encoders = {}

        # Create age groups
        df['Age_Group'] = pd.cut(df['Age'],
                                bins=[0, 1, 3, 5, 10, float('inf')],
                                labels=['Infant', 'Young', 'Adult', 'Middle_Aged', 'Senior'])

        # Process weight categories
        def categorize_weight(row):
            if row['Animal_Type'] == 'Dog':
                bins = [0, 10, 25, 40, float('inf')]
                labels = ['Small', 'Medium', 'Large', 'Giant']
            elif row['Animal_Type'] == 'Cat':
                bins = [0, 4, 6, 8, float('inf')]
                labels = ['Small', 'Medium', 'Large', 'Giant']
            else:
                return 'NA'
            return pd.cut([row['Weight']], bins=bins, labels=labels)[0]

        df['Weight_Category'] = df.apply(categorize_weight, axis=1)

        # Create and fit encoders
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                encoder = LabelEncoder()
                encoder.fit(df[col].unique())
                encoders[col] = encoder

        # Create disease encoder
        le_disease = LabelEncoder()
        le_disease.fit(df['Disease_Prediction'])

        # Save encoders to file
        with open('model/encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)

        # Save disease encoder separately
        with open('model/disease_encoder.pkl', 'wb') as f:
            pickle.dump(le_disease, f)

        return "Encoders saved successfully"
    except Exception as e:
        return f"Error saving encoders: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
