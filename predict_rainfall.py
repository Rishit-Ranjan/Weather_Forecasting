import pandas as pd
import joblib
import os

# Load the scaler, trained model, and model columns
if not all(os.path.exists(f) for f in ['scaler.pkl', 'rainfall_predictor_model.pkl', 'model_columns.pkl']):
    print("Error: Model files not found. Please run train_model.py first.")
    exit()

scaler = joblib.load('scaler.pkl')
model = joblib.load('rainfall_predictor_model.pkl')
model_columns = joblib.load('model_columns.pkl')

def predict_rainfall(new_data):
    """Predicts rainfall for new weather data."""
    new_df = pd.DataFrame([new_data])
    new_df = new_df.reindex(columns=model_columns, fill_value=0)
    features_to_scale = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'dayofweek']
    
    new_df[features_to_scale] = scaler.transform(new_df[features_to_scale])
    predicted_rainfall = model.predict(new_df)
    return predicted_rainfall[0]

def main():
    """Main function to demonstrate a prediction."""
    new_data = {
        'temperature': 22.5,
        'humidity': 60,
        'wind_speed': 3.0,
        'pressure': 1012,
        'hour': 14,
        'dayofweek': 2, # Wednesday
        'weather_scattered clouds': 1 
    }

    prediction = predict_rainfall(new_data)
    print(f"Predicted Rainfall (3-hour interval, mm): {prediction:.4f}")

if __name__ == "__main__":
    main()
