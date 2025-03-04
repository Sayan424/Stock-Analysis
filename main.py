from flask import Flask, jsonify, request
import requests
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = "O83XMS2GA8JNO283"  # Your Alpha Vantage API Key


@app.route('/')
def home():
    return "Stock Price API is running!"


# Get historical stock data
@app.route('/stock-history', methods=['GET'])
def get_stock_history():
    stock_symbol = request.args.get('symbol')

    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" in data:
        historical_data = data["Time Series (Daily)"]
        dates = []
        prices = []

        for date, values in historical_data.items():
            dates.append(date)
            prices.append(float(values["4. close"]))

        # Reverse the lists to have oldest first
        dates.reverse()
        prices.reverse()

        # Calculate volatility (Standard Deviation of daily returns)
        returns = np.diff(prices) / prices[:-1]  # Daily returns
        volatility = np.std(returns)

        # Calculate Simple Moving Average (SMA) for 10-day
        df = pd.DataFrame({"Date": dates, "Price": prices})
        df["SMA_10"] = df["Price"].rolling(window=10).mean()

        # Calculate Exponential Moving Average (EMA) for 10-day
        df["EMA_10"] = df["Price"].ewm(span=10, adjust=False).mean()

        return jsonify({
            "symbol": stock_symbol,
            "dates": dates,
            "prices": prices,
            "volatility": round(volatility, 5),
            "SMA_10": df["SMA_10"].dropna().tolist(),
            "EMA_10": df["EMA_10"].dropna().tolist()
        })
    else:
        return jsonify({
            "error": "Failed to fetch stock data",
            "details": data
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
