import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('cleaned_animal_disease_prediction.csv')

# Process data similar to training process
# Age groups
print("Creating age groups...")
df['Age_Group'] = pd.cut(df['Age'], 
                        bins=[0, 1, 3, 5, 10, float('inf')],
                        labels=['Infant', 'Young', 'Adult', 'Middle_Aged', 'Senior'])

# Weight categories
print("Categorizing weights...")
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

# Binary flags for symptoms
print("Creating symptom flags...")
symptom_cols = ['Appetite_Loss', 'Vomiting', 'Diarrhea', 'Coughing', 
               'Labored_Breathing', 'Lameness', 'Skin_Lesions', 
               'Nasal_Discharge', 'Eye_Discharge']
for col in symptom_cols:
    df[col] = (df[col] == 'Yes').astype(int)

# Generate encoders for categorical columns
print("Generating encoders...")
categorical_columns = ['Animal_Type', 'Breed', 'Gender', 'Age_Group', 'Weight_Category']
encoders = {}

# Create and fit encoders
for col in categorical_columns:
    if col in df.columns:
        print(f"Processing {col}...")
        df[col] = df[col].astype(str)
        encoder = LabelEncoder()
        encoder.fit(df[col].unique())
        encoders[col] = encoder
        print(f"  - {len(encoder.classes_)} unique classes")

# Create disease encoder
le_disease = LabelEncoder()
le_disease.fit(df['Disease_Prediction'])
print(f"Disease encoder: {len(le_disease.classes_)} unique diseases")
print("Disease classes:", ", ".join(le_disease.classes_))

# Save encoders to file
print("Saving encoders...")
with open('model/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Save disease encoder separately
with open('model/disease_encoder.pkl', 'wb') as f:
    pickle.dump(le_disease, f)

# Save mappings for reference
with open('model/class_mappings.txt', 'w') as f:
    f.write("Disease class mappings:\n")
    for i, disease in enumerate(le_disease.classes_):
        f.write(f"{i}: {disease}\n")
    
    f.write("\nCategory mappings:\n")
    for col, encoder in encoders.items():
        f.write(f"\n{col}:\n")
        for i, cls in enumerate(encoder.classes_):
            f.write(f"  {i}: {cls}\n")

print("Encoders saved successfully")
print("Done!")
