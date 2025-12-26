import os
import time
import datetime
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def fetch_weather_forecast(lat, lon, api_key, units='metric'):
    """Fetch forecast JSON from OpenWeatherMap (3-hourly forecast).
    Returns JSON dict.
    """
    base_url = 'http://api.openweathermap.org/data/2.5/forecast'
    params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': units}
    for attempt in range(3):
        resp = requests.get(base_url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            time.sleep(60)
        else:
            resp.raise_for_status()
    raise RuntimeError('Failed to fetch forecast after retries')


def parse_forecast_json(data):
    """Parse OpenWeatherMap forecast JSON into a DataFrame suitable for preprocessing.
    Keeps datetime, temperature, humidity, wind_speed, pressure, rainfall (3h) and weather description.
    """
    if 'list' not in data:
        raise ValueError('Unexpected forecast format')

    rows = []
    for entry in data['list']:
        dt = datetime.datetime.fromtimestamp(entry['dt'])
        main = entry.get('main', {})
        wind = entry.get('wind', {})
        rain = entry.get('rain', {}).get('3h', 0)
        weather = entry.get('weather', [{}])[0].get('description', '')

        rows.append({
            'datetime': dt,
            'temperature': main.get('temp'),
            'humidity': main.get('humidity'),
            'wind_speed': wind.get('speed', 0),
            'pressure': main.get('pressure'),
            'rainfall_3h': rain,
            'weather_desc': weather,
        })

    return pd.DataFrame(rows)


def preprocess_df(df):
    """Preprocess DataFrame: add hour/dayofweek and one-hot weather_desc.
    Returns processed DataFrame.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek

    # One-hot encode weather description
    dummies = pd.get_dummies(df['weather_desc'], prefix='weather', dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=['weather_desc'], inplace=True)

    return df
