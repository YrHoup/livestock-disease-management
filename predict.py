# Inference function that uses the saved components
import joblib
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
    preds = model.predict(X)
    return le.inverse_transform(preds)

if __name__ == "__main__":
    # Example usage
    data = {
         "Animal_Type": "Sheep",
          "Breed": "Suffolk",
          "Age": 3,
          "Gender": "Female",
          "Weight": 80.0,
          "Symptom_1": "Weight Loss",
          "Symptom_2": "Lethargy",
          "Symptom_3": "Appetite Loss",
          "Symptom_4": "Reduced Wool Growth",
          "Duration": "8 days",
          "Appetite_Loss": "Yes",
          "Vomiting": "No",
          "Diarrhea": "Yes",
          "Coughing": "No",
          "Labored_Breathing": "No",
          "Lameness": "Yes",
          "Skin_Lesions": "No",
          "Nasal_Discharge": "Yes",
          "Eye_Discharge": "No",
          "Body_Temperature": "39.8°C",
          "Heart_Rate": 77
    }
    result = predict_disease(data)
    print(f"Predicted Disease: {result}")
