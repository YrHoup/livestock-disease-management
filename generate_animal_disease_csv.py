import pandas as pd
import numpy as np

# Define possible values for categorical columns
animal_names = ['Dog', 'Cat', 'Cow', 'Goat', 'Sheep', 'Pig', 'Horse', 'Rabbit', 'Chicken', 'Duck']
symptoms = [
    'Fever', 'Cough', 'Lethargy', 'Diarrhea', 'Vomiting', 'Loss of appetite',
    'Difficulty breathing', 'Lameness', 'Swelling', 'Skin rash'
]
dangerous = ['Yes', 'No']

# Generate synthetic data mimicking the Animal Disease dataset
np.random.seed(42)  # For reproducibility
n_samples = 1000

data = {
    'AnimalName': np.random.choice(animal_names, n_samples),
    'symptoms1': np.random.choice(symptoms, n_samples),
    'symptoms2': np.random.choice(symptoms, n_samples),
    'symptoms3': np.random.choice(symptoms, n_samples),
    'symptoms4': np.random.choice(symptoms, n_samples),
    'symptoms5': np.random.choice(symptoms, n_samples),
    'Dangerous': np.random.choice(dangerous, n_samples, p=[0.3, 0.7])  # Slightly imbalanced
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('animal_disease_data.csv', index=False)
print("CSV file 'animal_disease_data.csv' created successfully.")