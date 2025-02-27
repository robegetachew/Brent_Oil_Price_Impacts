from flask import Flask, jsonify, request
from flask_cors import CORS
from src.data import BrentDataLoader
from src.ml_model import LSTMOilPredictor
from src.models import OilPriceModels
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize data and models
data_loader = BrentDataLoader('data/merged_brent_events.csv')
data = data_loader.preprocess_data()
lstm_predictor = LSTMOilPredictor(data)
oil_models = OilPriceModels(data)

@app.route('/api/data', methods=['GET'])
def get_data():
    """Return preprocessed Brent oil data."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    df = data
    if start_date and end_date:
        df = df.loc[start_date:end_date]
    return jsonify(df.reset_index().to_dict(orient='records'))

@app.route('/api/lstm_forecast', methods=['GET'])
def get_lstm_forecast():
    """Return LSTM forecast for a given period."""
    X, _ = lstm_predictor.prepare_data()
    lstm_predictor.build_model()
    lstm_predictor.train_model(X, X[:, -1], epochs=10)  # Simplified training for demo
    forecast = lstm_predictor.predict(X[-252:])  # Last year forecast
    dates = data.index[-252:].strftime('%Y-%m-%d').tolist()
    return jsonify({'dates': dates, 'forecast': forecast.flatten().tolist()})

@app.route('/api/arima_forecast', methods=['GET'])
def get_arima_forecast():
    """Return ARIMA forecast."""
    arima_results = oil_models.fit_arima_with_events()
    forecast = arima_results.forecast(steps=252)
    dates = pd.date_range(start=data.index[-1], periods=252, freq='D').strftime('%Y-%m-%d').tolist()
    return jsonify({'dates': dates, 'forecast': forecast.tolist()})

@app.route('/api/volatility', methods=['GET'])
def get_volatility():
    """Return GARCH volatility."""
    garch_results = oil_models.fit_garch()
    volatility = garch_results.conditional_volatility
    dates = data.index[1:].strftime('%Y-%m-%d').tolist()  # Skip NaN from Returns
    return jsonify({'dates': dates, 'volatility': volatility.tolist()})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics."""
    X, y = lstm_predictor.prepare_data()
    lstm_predictor.build_model()
    lstm_predictor.train_model(X, y, epochs=10)
    preds = lstm_predictor.predict(X)
    actual = lstm_predictor.scaler.inverse_transform(y.reshape(-1, 1))
    metrics = oil_models.evaluate_predictions(actual, preds)
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)