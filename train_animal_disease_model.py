import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# Load the dataset
df = pd.read_csv('animal_disease_data.csv')

# Basic EDA
print("Dataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Visualize distribution of Dangerous column
plt.figure(figsize=(6, 4))
sns.countplot(x='Dangerous', data=df)
plt.title('Distribution of Dangerous Animals')
plt.savefig('dangerous_distribution.png')
plt.close()

# Visualize symptom frequency
symptom_cols = ['symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5']
symptom_counts = pd.Series(df[symptom_cols].values.ravel()).value_counts()
plt.figure(figsize=(10, 6))
symptom_counts.plot(kind='bar')
plt.title('Frequency of Symptoms')
plt.xlabel('Symptom')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('symptom_frequency.png')
plt.close()

# Preprocessing
# Encode categorical variables
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Split features and target
X = df.drop('Dangerous', axis=1)
y = df['Dangerous']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance
plt.figure(figsize=(8, 6))
xgb.plot_importance(xgb_model)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save the model
xgb_model.save_model('animal_disease_xgb_model.json')
print("Model saved as 'animal_disease_xgb_model.json'.")