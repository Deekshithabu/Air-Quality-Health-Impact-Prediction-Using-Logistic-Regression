from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)
app.secret_key = 'devkey'

MODEL_PATH = "model/logistic_model.pkl"
SCALER_PATH = "model/scaler.pkl"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    # Train model from CSV placed in data/air_quality.csv
    csv_path = 'data/air_quality.csv'
    if not os.path.exists(csv_path):
        return 'ERROR: data/air_quality.csv not found. Place your dataset there and retry.'
    df = pd.read_csv(csv_path)
    # Keep only pollutant columns if present
    required = ["PM2.5","PM10","NO2","SO2","CO","O3"]

    df = df.dropna(subset=required)
    # If AQI not present, compute as mean of pollutants as a quick proxy
    if 'AQI' not in df.columns:
        df['AQI'] = df[required].mean(axis=1)
    df['Health_Risk'] = (df['AQI'] > 100).astype(int)

    X = df[required]
    y = df['Health_Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return '✅ Model trained and saved to model/ (logistic_model.pkl and scaler.pkl)'

@app.route('/predict', methods=['POST'])
def predict():
    required = ["PM2.5","PM10","NO2","SO2","CO","O3"]

    # Load model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return 'ERROR: Model not trained. Visit /train to train the model first.'

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    try:
        values = [float(request.form.get(c)) for c in required]
    except Exception as e:
        return f'ERROR: invalid input values. {e}'

    scaled = scaler.transform([values])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0,1] if hasattr(model, 'predict_proba') else None
    return render_template('index.html', result={'pred': int(pred), 'prob': float(prob) if prob is not None else None})

if __name__ == '__main__':
    app.run(debug=True)
