
import os
import requests
import pandas as pd
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# It's recommended to store API keys securely, e.g., as environment variables, rather than hardcoding.
API_KEY = os.getenv("OPENWEATHER_API_KEY", "Place_your_api_key_here") # Fallback for convenience
LATITUDE = 23.12
LONGITUDE = 77.05
BASE_URL = f'http://api.openweathermap.org/data/2.5/forecast?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}&units=metric'

# Define base directory for saving artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def fetch_weather_data(url):
    """Fetches weather data from the OpenWeatherMap API."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()

def parse_forecast_data(data):
    """Parses the JSON response from the API into a pandas DataFrame."""
    if 'list' not in data:
        raise ValueError(f"Unexpected response format: {data}")
    
    all_rows = []
    for entry in data['list']:
        dt = datetime.datetime.fromtimestamp(entry['dt'])
        temp = entry['main']['temp']
        humidity = entry['main']['humidity']
        wind_speed = entry['wind']['speed']
        pressure = entry['main']['pressure']
        rainfall = entry.get('rain', {}).get('3h', 0)
        weather_desc = entry['weather'][0]['description']

        row = {
            'datetime': dt,
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'pressure': pressure,
            'rainfall_3h': rainfall,
            'weather_desc': weather_desc,
        }
        all_rows.append(row)
        
    return pd.DataFrame(all_rows)

def preprocess_data(df):
    """Preprocesses the raw weather data for model training."""
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # One-hot encode weather description
    weather_dummies = pd.get_dummies(df['weather_desc'], prefix='weather', dtype=int)
    df = pd.concat([df, weather_dummies], axis=1)
    df.drop('weather_desc', axis=1, inplace=True)
    
    return df

def main():
    """Main function to run the training pipeline."""
    if API_KEY == "Place_your_api_key_here":
        print("Warning: Using placeholder API key. Please set OPENWEATHER_API_KEY environment variable.")

    print("Fetching weather data...")
    try:
        raw_data = fetch_weather_data(BASE_URL)
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return

    df = parse_forecast_data(raw_data)
    csv_path = os.path.join(BASE_DIR, 'weather_data.csv')
    df.to_csv(csv_path, index=False)
    print("Weather data saved to weather_data.csv")

    print("Preprocessing data...")
    # Use the dataframe directly to avoid redundant I/O
    df = preprocess_data(df)

    # Prepare features and target
    X = df.drop(['datetime', 'rainfall_3h'], axis=1)
    y = df['rainfall_3h']

    # Save the feature names for the prediction step
    model_columns = X.columns.tolist()
    joblib.dump(model_columns, os.path.join(BASE_DIR, 'model_columns.pkl'))
    print("Model columns saved to model_columns.pkl")

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Numerical Features (Fit on training data only to prevent data leakage)
    features_to_scale = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'dayofweek']
    scaler = StandardScaler()
    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

    # Save the scaler
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    print("Scaler saved to scaler.pkl")

    print("Training the model...")
    # Initialize and Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make Predictions on Test Data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Save the Trained Model
    joblib.dump(model, os.path.join(BASE_DIR, 'rainfall_predictor_model.pkl'))
    print("Model saved to rainfall_predictor_model.pkl")

if __name__ == "__main__":
    main()
