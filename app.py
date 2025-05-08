import json
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, render_template, request, redirect, session, flash

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


if __name__ == '__main__':
    app.run(debug=True)
