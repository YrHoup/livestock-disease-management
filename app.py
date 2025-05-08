import json
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, redirect, session, flash, jsonify

from predict import predict_disease, check_healthy
from rsa import generate_rsa_keys, rsa_encrypt

public_key, private_key = generate_rsa_keys()
app = Flask(__name__, static_folder='static', static_url_path='/static')
DATABASE = 'livestock.db'
app.secret_key = 'your-very-secret-key'

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


def get_db():
    """Connect to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Allows us to access rows as dictionaries
    return conn


def init_db():
    """Initialize the database with a user table."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                atype TEXT UNIQUE NOT NULL
            )
        """)
        conn.execute("""
                CREATE TABLE IF NOT EXISTS breeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type_id INTEGER NOT NULL,
                    breed TEXT UNIQUE NOT NULL,
                    FOREIGN KEY (type_id) REFERENCES types(id) ON DELETE CASCADE
                )
        """)


init_db()


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
            temp_data = {
                "Animal_Type": request.form['Animal_Type'],
                "Breed": request.form['Breed'],
                "Age": int(request.form['Age']),
                "Gender": request.form['Gender'],
                "Weight": float(request.form['Weight']),
                "Symptom_1": request.form['Symptom_1'],
                "Symptom_2": request.form['Symptom_2'],
                "Symptom_3": request.form['Symptom_3'],
                "Symptom_4": request.form['Symptom_4'],
                "Duration": request.form['Duration'],
                "Body_Temperature": request.form['Body_Temperature'],
                "Heart_Rate": int(request.form['Heart_Rate']),
            }

            binary_data = {
                'Appetite Loss': 'Appetite_Loss',
                'Vomiting': 'Vomiting',
                'Diarrhea': 'Diarrhea',
                'Coughing': 'Coughing',
                'Labored Breathing': 'Labored_Breathing',
                'Lameness': 'Lameness',
                'Skin Lesions': 'Skin_Lesions',
                'Nasal Discharge': 'Nasal_Discharge',
                'Eye Discharge': 'Eye_Discharge'
            }

            for col in binary_data.values():
                temp_data[col] = 'No'

            for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
                symptom = temp_data[col]
                if symptom in binary_data:
                    temp_data[binary_data[symptom]] = 'Yes'

            ordered_keys = [
                "Animal_Type", "Breed", "Age", "Gender", "Weight", "Symptom_1", "Symptom_2",
                "Symptom_3", "Symptom_4",
                "Duration", "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing",
                "Labored_Breathing", "Lameness", "Skin_Lesions",
                "Nasal_Discharge", "Eye_Discharge", "Body_Temperature", "Heart_Rate"
            ]

            data = {key: temp_data.get(key) for key in ordered_keys}

            checked_data = check_healthy(data)

            if checked_data['is_healthy']:

                prediction = {
                    'Disease': "Healthy",
                    'Confidence': checked_data['Confidence']
                }
                advice = disease_advice[0]
            else:
                # Make prediction
                prediction = predict_disease(data)

                # Generate more specific advice based on symptoms
                symptoms = [data[f'Symptom_{i}'] for i in range(1, 4) if data[f'Symptom_{i}'] != 'No']
                advice = generate_advice(prediction['Disease'] != "Healthy", data['Animal_Type'], symptoms)

            return render_template('predict.html',
                                   prediction=prediction,
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


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not username or not password:
            flash('Both username and password are required!', 'danger')
            return redirect('/signup')

        # Check if the username already exists
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username already taken. Please choose another one.', 'danger')
            return redirect('/signup')

        encrypted_password = json.dumps(rsa_encrypt(public_key, password))

        print(encrypted_password)
        # Insert the new user into the database
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, encrypted_password))
        conn.commit()
        flash('User successfully registered!', 'success')
        session['username'] = username
        flash('User successfully registered and logged in!', 'success')
        return redirect('/')

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if not user:
            flash('User not found. Please try again.', 'danger')
        else:
            encrypted_password = json.dumps(rsa_encrypt(public_key, password))
            if user['password'] == encrypted_password:
                session['username'] = username
                flash('Login successful!', 'success')
                return redirect('/')
            else:
                flash('Incorrect password. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

@app.route('/get_animal_types')
def get_animal_types():
    conn = sqlite3.connect('livestock.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT atype FROM types")
    types = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(types)


@app.route('/get_breeds/<animal_type>')
def get_breeds(animal_type):
    conn = sqlite3.connect('livestock.db')
    cursor = conn.cursor()

    cursor.execute("""
                   SELECT b.breed
                   FROM breeds b
                            JOIN types a ON b.type_id = a.id
                   WHERE a.atype = ?
                   """, (animal_type,))

    breeds = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify(breeds)


if __name__ == '__main__':
    app.run(debug=True)
