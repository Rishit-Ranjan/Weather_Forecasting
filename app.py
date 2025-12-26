
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import os
from dotenv import load_dotenv
from model_module import predict_from_model
from data_processing import fetch_weather_forecast, parse_forecast_json

app = Flask(__name__)

# Define base directory to ensure files are found
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment
load_dotenv(os.path.join(BASE_DIR, '.env'))
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', "98978729593a6890d12daaf6e3492d66")

# Check model artifacts exist (model_module will also load them when predicting)
missing = [p for p in ['rainfall_predictor_model.pkl','scaler.pkl','model_columns.pkl'] if not os.path.exists(os.path.join(BASE_DIR,p))]
if missing:
    print(f"Warning: Missing model artifacts: {missing}. Run training first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        # Allow two modes: fetch by lat/lon OR manual feature input
        if data is None:
            return jsonify({'error':'Invalid JSON payload'}), 400

        input_payload = None
        # If lat/lon provided, fetch forecast and use the first forecast entry
        if 'lat' in data and 'lon' in data:
            if not OPENWEATHER_API_KEY:
                return jsonify({'error':'Server missing OPENWEATHER_API_KEY'}), 500
            # validate lat/lon
            try:
                lat_val = float(data['lat'])
                lon_val = float(data['lon'])
            except (TypeError, ValueError):
                return jsonify({'error':'Invalid latitude/longitude values'}), 400
            if not (-90.0 <= lat_val <= 90.0 and -180.0 <= lon_val <= 180.0):
                return jsonify({'error':'Latitude or longitude out of range'}), 400

            units = data.get('units','metric')
            forecast = fetch_weather_forecast(lat_val, lon_val, OPENWEATHER_API_KEY, units=units)
            df = parse_forecast_json(forecast)
            if df.empty:
                return jsonify({'error':'No forecast data returned from weather API'}), 500

            # pick first forecast entry that has required numeric fields
            required_numeric = ['temperature', 'humidity', 'wind_speed', 'pressure']
            good_rows = df.dropna(subset=required_numeric)
            if good_rows.empty:
                return jsonify({'error':'Forecast entries missing required numeric fields'}), 500

            row = good_rows.iloc[0]

            # ensure values are valid numbers
            try:
                temp = float(row['temperature'])
                hum = float(row['humidity'])
                wind = float(row['wind_speed'])
                pres = float(row['pressure'])
            except (TypeError, ValueError) as e:
                return jsonify({'error':'Forecast contains non-numeric weather values'}), 500

            input_payload = {
                'temperature': temp,
                'humidity': hum,
                'wind_speed': wind,
                'pressure': pres,
                'datetime': row['datetime'].isoformat(),
                'weather_desc': row['weather_desc'] if row.get('weather_desc') is not None else ''
            }
        else:
            # Expect manual fields
            required = ['temperature','humidity','wind_speed','pressure','datetime','weather_desc']
            missing = [k for k in required if k not in data]
            if missing:
                return jsonify({'error':f'Missing fields: {missing}'}), 400
            # Validate and coerce manual inputs
            try:
                temp = float(data['temperature'])
                hum = float(data['humidity'])
                wind = float(data['wind_speed'])
                pres = float(data['pressure'])
            except (TypeError, ValueError):
                return jsonify({'error':'Manual input numeric fields must be valid numbers'}), 400

            # validate datetime
            try:
                _ = pd.to_datetime(data['datetime'])
            except Exception:
                return jsonify({'error':'Invalid datetime format'}), 400

            input_payload = {
                'temperature': temp,
                'humidity': hum,
                'wind_speed': wind,
                'pressure': pres,
                'datetime': data['datetime'],
                'weather_desc': data.get('weather_desc','')
            }

        # Use model_module wrapper to get prediction in mm
        try:
            pred_mm = predict_from_model(input_payload)
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error':str(e)}), 500

        # Convert mm to inches
        mm_to_in = 0.0393701
        pred_in = pred_mm * mm_to_in

        return jsonify({'prediction_mm': round(pred_mm,4), 'prediction_inches': round(pred_in,4)})
        
    except Exception as e:
        print(f"Unhandled error in /predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
