
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Define base directory to ensure files are found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model and artifacts
try:
    model = joblib.load(os.path.join(BASE_DIR, 'rainfall_predictor_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        
        # Debug: Print received data to console
        print(f"Received data: {data}")

        # Check for required keys to prevent KeyError
        required_keys = ['temperature', 'humidity', 'wind_speed', 'pressure', 'datetime', 'weather_desc']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({'error': f"Missing keys: {', '.join(missing_keys)}"}), 400
        
        # Parse inputs
        features_dict = {
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'wind_speed': float(data['wind_speed']),
            'pressure': float(data['pressure']),
        }
        
        # Process datetime to get hour and dayofweek
        dt = pd.to_datetime(data['datetime'])
        features_dict['hour'] = dt.hour
        features_dict['dayofweek'] = dt.dayofweek
        
        # Scale numerical features (Order must match training)
        scale_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'dayofweek']
        df_to_scale = pd.DataFrame([features_dict])[scale_cols]
        scaled_values = scaler.transform(df_to_scale)
        
        # Prepare final dataframe with all model columns initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Fill scaled numerical values
        input_df.loc[0, scale_cols] = scaled_values[0]
        
        # Handle One-Hot Encoding for weather description
        weather_col = f"weather_{data['weather_desc']}"
        if weather_col in input_df.columns:
            input_df.loc[0, weather_col] = 1
        
        # Make a prediction using the dataframe
        prediction = model.predict(input_df)
        
        # Prepare the response
        # The model is LinearRegression, so it predicts rainfall amount in mm, not Yes/No
        output = f"{prediction[0]:.2f} mm"
        
        return jsonify({'prediction': output})
        
    except Exception as e:
        print(f"Prediction error: {e}") # Log error to console
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
