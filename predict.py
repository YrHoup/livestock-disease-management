# Inference function that uses the saved components
import joblib
import sqlite3
import numpy as np
import pandas as pd


def predict_disease(new_data):
    # Load artifacts
    scaler = joblib.load("model/scaler.joblib")
    encoders = joblib.load("model/encoders.joblib")
    selector = joblib.load("model/selector.joblib")
    model = joblib.load("model/animal_disease_xgb_model.joblib")
    le = joblib.load("model/label_encoder.joblib")
    training_columns = joblib.load("model/training_columns.joblib")
    age_params = joblib.load("model/age_params.joblib")
    weight_params = joblib.load("model/weight_params.joblib")
    original_features = joblib.load("model/original_features.joblib")
    constant_filter = joblib.load("model/constant_filter.joblib")

    print(new_data)

    df = pd.DataFrame([new_data])

    # Binary symptom flags
    symptom_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing',
                    'Labored_Breathing', 'Lameness', 'Skin_Lesions',
                    'Nasal_Discharge', 'Eye_Discharge']
    for col in symptom_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

    # Temperature conversion
    df['Body_Temperature'] = df['Body_Temperature'].str.replace('°C', '').astype(float)

    # Duration conversion
    df['Duration_Num'] = df['Duration'].str.extract(r'(\d+)').astype(float)
    df.loc[df['Duration'].str.contains('week'), 'Duration_Num'] *= 7

    # Age groups
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=age_params["bins"],
                             labels=age_params["labels"])

    # Weight categories
    def categorize_weight(row):
        animal_type = row['Animal_Type']
        if animal_type in weight_params:
            bins = weight_params[animal_type]["bins"]
            labels = weight_params[animal_type]["labels"]
            return pd.cut([row['Weight']], bins=bins, labels=labels)[0]
        return 'NA'

    df['Weight_Category'] = df.apply(categorize_weight, axis=1)

    # Replace symptom names
    symptom_cols = [col for col in df if col.startswith('Symptom_')]
    df[symptom_cols] = df[symptom_cols].replace('Loss of Appetite', 'Appetite Loss')

    # Encode categorical features
    cat_features = ['Animal_Type', 'Breed', 'Gender', 'Age_Group', 'Weight_Category']
    for feature in cat_features:
        encoder = encoders.get(feature)
        if encoder:
            df[feature] = df[feature].astype(str).apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
            )

    # One-hot encode remaining objects
    obj_cols = df.select_dtypes('object').columns
    df = pd.get_dummies(df, columns=obj_cols)

    # Align columns with training data
    df = df.reindex(columns=training_columns, fill_value=0)

    # Feature scaling
    num_features = ['Age', 'Weight', 'Body_Temperature', 'Duration_Num', 'Heart_Rate']
    df[num_features] = scaler.transform(df[num_features])

    df = df.reindex(columns=original_features, fill_value=0)

    # Now apply variance threshold (101 -> 79 features)
    df = df.loc[:, constant_filter.get_support()]

    # Continue with feature selection and prediction
    X = selector.transform(df)
    # Get prediction probabilities
    pred_proba = model.predict_proba(X)

    # Get predicted class index
    preds = model.predict(X)

    # Get confidence score (probability of predicted class)
    confidence = np.max(pred_proba, axis=1)

    prediction = {
        'Disease': le.inverse_transform(preds)[0],
        'Confidence': f"{confidence[0] * 100:.2f}%"
    }

    print(prediction)

    return prediction


def load_animal_ranges_from_db():
    conn = sqlite3.connect('livestock.db')
    cursor = conn.cursor()

    cursor.execute("SELECT atype, weight_begin, weight_end, temp_begin, temp_end, heart_rate_begin, heart_rate_end FROM types")
    rows = cursor.fetchall()
    conn.close()

    animal_ranges = {}
    for row in rows:
        animal_type = row[0]
        animal_ranges[animal_type] = {
            'weight_range': (row[1], row[2]),
            'temp_range': (row[3], row[4]),
            'rate_range': (row[5], row[6])
        }
    return animal_ranges

def check_healthy(data):
    symptom_values = [data['Symptom_1'], data['Symptom_2'], data['Symptom_3'], data['Symptom_4']]
    animal_ranges = load_animal_ranges_from_db()
    animal_type = data['Animal_Type']

    # Counts the number of symptoms that are not 'No'
    symptom_count = sum(1 for symptom in symptom_values if symptom == 'No')
    confidence = 100.0
    disease = 'Healthy'
    is_healthy = True

    if animal_type in animal_ranges:
        weight_range = animal_ranges[animal_type]['weight_range']
        temp_range = animal_ranges[animal_type]['temp_range']
        rate_range = animal_ranges[animal_type]['rate_range']

        # Symptom impact
        if symptom_count == 4:
            pass
        elif symptom_count == 3:
            confidence -= 40.0
        else:
            confidence -= 60.0

        # Weight check
        weight = float(data['Weight'])
        if not (weight_range[0] <= weight <= weight_range[1]):
            confidence -= 30.0

        # Temperature check
        temp = float(data['Body_Temperature'].replace("°C", "").strip())
        if not (temp_range[0] <= temp <= temp_range[1]):
            confidence -= 30.0

        # Heart rate check
        rate = int(data['Heart_Rate'])
        if not (rate_range[0] <= rate <= rate_range[1]):
            confidence -= 30.0

    if confidence < 50.0:
        is_healthy = False
        disease = 'Unhealthy'

    prediction = {
        'is_healthy': is_healthy,
        'Disease': disease,
        'Confidence': f"{confidence:.2f}%"
    }

    return prediction


if __name__ == "__main__":
    # Example usage
    data = {
  "Animal_Type": "Cow",
  "Breed": "Angus",
  "Age": 5,
  "Gender": "Female",
  "Weight": 700.0,
  "Symptom_1": "Fever",
  "Symptom_2": "Decreased Milk Yield",
  "Symptom_3": "Lethargy",
  "Symptom_4": "Nasal Discharge",
  "Duration": "7 days",
  "Appetite_Loss": "Yes",
  "Vomiting": "No",
  "Diarrhea": "No",
  "Coughing": "Yes",
  "Labored_Breathing": "Yes",
  "Lameness": "No",
  "Skin_Lesions": "No",
  "Nasal_Discharge": "Yes",
  "Eye_Discharge": "No",
  "Body_Temperature": "39.8°C",
  "Heart_Rate": 80
}
    result = predict_disease(data)
    print(f"Predicted Disease: {result}")

