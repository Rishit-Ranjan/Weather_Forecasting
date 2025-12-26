
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('rainfall_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        
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
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
