import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Alpha Vantage API key and URL
API_KEY = 'I7GGWW15MGHJ44MX'  # Replace with your actual API key
BASE_URL = 'https://www.alphavantage.co/query'

# Function to fetch live stock data (intraday data)
def get_live_data(symbol, interval='1min', data_type='json'):
    url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}&datatype={data_type}"
    response = requests.get(url)
    data = response.json()
    if 'Time Series (1min)' in data:
        return pd.DataFrame(data['Time Series (1min)']).T
    else:
        return None

# Function to fetch historical stock data (daily data)
def get_stock_data(ticker='TSLA'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}&outputsize=compact'
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        return None, "Error: Could not fetch data or invalid ticker."

    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df = df[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.astype(float)

    return df, None

# Function to predict stock price using linear regression (daily data)
def predict_stock(ticker='TSLA'):
    df, error = get_stock_data(ticker)
    if error:
        return None, error

    df['Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Next_Close']

    model = LinearRegression()
    model.fit(X, y)

    latest_features = X.iloc[-1].values.reshape(1, -1)
    predicted_close = model.predict(latest_features)[0]

    df['Predicted_Close'] = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Actual Close', color='blue')
    ax.plot(df.index, df['Predicted_Close'], label='Predicted Close', color='orange')
    ax.set_title(f'{ticker} Actual vs Predicted Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return {
        'predicted_price': round(predicted_close, 2),
        'graph': plot_url
    }, None

# Function to predict stock price using linear regression (intraday data)
def predict_stock_price(symbol):
    live_data = get_live_data(symbol)
    if live_data is not None:
        # Process and use this live data for prediction
        live_data['close'] = pd.to_numeric(live_data['4. close'])
        X = np.array(range(len(live_data))).reshape(-1, 1)
        y = live_data['close'].values
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([[len(live_data) + 1]])  # Predict next minute's closing price
        return prediction[0]
    else:
        return "Error: No data found for the ticker."

# Function to generate prediction graph (returning as base64 image) for intraday data
def generate_prediction_graph(symbol):
    live_data = get_live_data(symbol)
    if live_data is not None:
        live_data.index = pd.to_datetime(live_data.index)  # Ensure datetime index
        live_data['close'] = pd.to_numeric(live_data['4. close'])

        X = np.array(range(len(live_data))).reshape(-1, 1)
        y = live_data['close'].values
        model = LinearRegression()
        model.fit(X, y)

        predicted_price = model.predict([[len(live_data) + 1]])  # Predict next step

        # Create future index
        future_time = live_data.index[-1] + pd.Timedelta(minutes=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(live_data.index, live_data['close'], label='Actual Price')
        plt.plot(
            list(live_data.index) + [future_time],
            list(y) + [predicted_price[0]],
            label='Predicted Price',
            linestyle='dashed',
            color='orange'
        )
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.title(f'{symbol} - Actual vs Predicted Stock Prices')
        plt.legend()
        plt.tight_layout()

        # Save to memory and encode
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{graph_url}", None
    return None, "Error: Could not retrieve data."
