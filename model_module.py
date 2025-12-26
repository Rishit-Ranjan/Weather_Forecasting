import os
import joblib
import pandas as pd
from data_processing import preprocess_df, parse_forecast_json, fetch_weather_forecast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_from_dataframe(df):
    """Train LinearRegression model from a preprocessed DataFrame and save artifacts."""
    # Import scikit-learn here to avoid heavy imports when the module is imported by the web app
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    df = preprocess_df(df)

    X = df.drop(['datetime', 'rainfall_3h'], axis=1)
    y = df['rainfall_3h']

    model_columns = X.columns.tolist()
    joblib.dump(model_columns, os.path.join(BASE_DIR, 'model_columns.pkl'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    features_to_scale = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'dayofweek']
    scaler = StandardScaler()
    X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Trained model MSE={mse:.4f}, R2={r2:.4f}")

    joblib.dump(model, os.path.join(BASE_DIR, 'rainfall_predictor_model.pkl'))
    return model


def predict_from_model(input_dict):
    """Load artifacts and predict rainfall (returns mm)."""
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    model = joblib.load(os.path.join(BASE_DIR, 'rainfall_predictor_model.pkl'))
    model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

    new_df = pd.DataFrame([input_dict])
    # ensure hour/dayofweek present
    if 'datetime' in new_df.columns:
        new_df['datetime'] = pd.to_datetime(new_df['datetime'])
        new_df['hour'] = new_df['datetime'].dt.hour
        new_df['dayofweek'] = new_df['datetime'].dt.dayofweek

    # One-hot alignment
    new_df = new_df.reindex(columns=model_columns, fill_value=0)

    features_to_scale = ['temperature', 'humidity', 'wind_speed', 'pressure', 'hour', 'dayofweek']
    new_df[features_to_scale] = scaler.transform(new_df[features_to_scale])

    pred = model.predict(new_df)
    return float(pred[0])
