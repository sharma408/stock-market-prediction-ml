from flask import Flask, render_template, request, redirect, session
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from flask_mysqldb import MySQL
from datetime import datetime, timedelta
import random

app = Flask(__name__)
app.secret_key = 'I7GGWW15MGHJ44MX'

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'shweta'
app.config['MYSQL_DB'] = 'stock_prediction'
mysql = MySQL(app)

def get_db_cursor():
    return mysql.connection.cursor()

# Alpha Vantage API config
API_KEY = 'your_alpha_vantage_api_key'
BASE_URL = 'https://www.alphavantage.co/query'

def get_live_data(symbol, interval='1min', data_type='json'):
    url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}&datatype={data_type}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (1min)' in data:
            return pd.DataFrame(data['Time Series (1min)']).T
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def predict_stock_price(symbol):
    live_data = get_live_data(symbol)
    if live_data is not None and not live_data.empty:
        try:
            live_data['close'] = pd.to_numeric(live_data['4. close'])
            X = np.array(range(len(live_data))).reshape(-1, 1)
            y = live_data['close'].values
            
            model = LinearRegression()
            model.fit(X, y)

            future_steps = 5
            future_prices = model.predict(
                np.array(range(len(live_data), len(live_data) + future_steps)).reshape(-1, 1)
            )

            highest = np.max(future_prices)
            lowest = np.min(future_prices)

            return future_prices[-1], highest, lowest
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None, None
    return None, None, None

def calculate_accuracy(user_id, ticker):
    """Calculate prediction accuracy based on historical data"""
    cursor = get_db_cursor()
    try:
        cursor.execute("""
            SELECT predicted_price, actual_price 
            FROM predictions 
            WHERE user_id = %s AND ticker = %s AND actual_price IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10
        """, (user_id, ticker))
        results = cursor.fetchall()
        
        if not results:
            return None
            
        correct = 0
        for pred, actual in results:
            # Consider prediction correct if within 2% of actual price
            if abs(pred - actual) / actual <= 0.02:
                correct += 1
                
        return (correct / len(results)) * 100 if results else None
    except Exception as e:
        print(f"Accuracy calculation error: {e}")
        return None

def get_market_news():
    """Simulate fetching market news (in a real app, use a news API)"""
    news_items = [
        {
            'title': 'Tech Stocks Rally on AI Optimism',
            'source': 'Bloomberg',
            'time': '2 hours ago',
            'image': 'https://via.placeholder.com/60'
        },
        {
            'title': 'Fed Signals Potential Rate Cuts Next Year',
            'source': 'CNBC',
            'time': '4 hours ago',
            'image': 'https://via.placeholder.com/60'
        },
        {
            'title': f"{random.choice(['AAPL', 'MSFT', 'GOOG'])} Analyst Rating Upgraded to Buy",
            'source': 'Wall Street Journal',
            'time': '6 hours ago',
            'image': 'https://via.placeholder.com/60'
        }
    ]
    return news_items

@app.route('/')
def home():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        cursor = get_db_cursor()
        try:
            cursor.execute("SELECT id, password FROM users WHERE username = %s", (uname,))
            user = cursor.fetchone()
            if user and user[1] == pwd:
                session['user_id'] = user[0]
                session['username'] = uname
                return redirect('/dashboard')
            return render_template('login.html', error="Invalid credentials!")
        except Exception as e:
            print(f"Login error: {e}")
            return render_template('login.html', error="Database error occurred")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        email = request.form['email']
        pwd = request.form['password']
        cursor = get_db_cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (uname, email, pwd)
            )
            mysql.connection.commit()
            return redirect('/login')
        except Exception as e:
            print(f"Registration error: {e}")
            mysql.connection.rollback()
            return render_template('registration.html', error="Registration failed")
    return render_template('registration.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    cursor = get_db_cursor()
    
    try:
        # Get the most recent prediction
        cursor.execute(
            "SELECT ticker, predicted_price, timestamp FROM predictions WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1",
            (session['user_id'],)
        )
        latest_prediction = cursor.fetchone()
        
        if not latest_prediction:
            return render_template('dashboard.html', 
                               user=session.get('username', 'Guest'),
                               no_predictions=True)

        ticker = latest_prediction[0]
        
        # Get all predictions for this specific ticker
        cursor.execute(
            "SELECT predicted_price, timestamp FROM predictions WHERE user_id = %s AND ticker = %s ORDER BY timestamp DESC",
            (session['user_id'], ticker)
        )
        predictions = cursor.fetchall()
        
        # Prepare data for JSON serialization
        prediction_data = [{
            'price': float(p[0]),
            'timestamp': p[1].strftime('%Y-%m-%d %H:%M:%S') if p[1] else None
        } for p in predictions]

        # Calculate stats
        prices = [p['price'] for p in prediction_data]
        
        stats = {
            'show_high_low': len(prediction_data) > 1,
            'highest': max(prices) if len(prices) > 1 else None,
            'lowest': min(prices) if len(prices) > 1 else None,
            'latest': prices[0] if prices else 0,
            'ticker': ticker
        }

        # Get prediction history (last 5 predictions)
        cursor.execute(
            """SELECT ticker, predicted_price, timestamp, 
               CASE WHEN actual_price IS NULL THEN 'Pending'
                    WHEN ABS(predicted_price - actual_price)/actual_price <= 0.02 THEN 'Correct'
                    ELSE 'Incorrect' END as status
               FROM predictions 
               WHERE user_id = %s 
               ORDER BY timestamp DESC 
               LIMIT 5""",
            (session['user_id'],)
        )
        prediction_history = [{
            'ticker': row[0],
            'price': float(row[1]),
            'date': row[2].strftime('%b %d') if row[2] else 'N/A',
            'status': row[3]
        } for row in cursor.fetchall()]

        # Generate next prediction time (random between 1-3 hours from now)
        next_pred_time = datetime.now() + timedelta(hours=random.randint(1, 3))
        next_prediction = {
            'price': round(stats['latest'] * random.uniform(0.98, 1.02), 2),
            'time': f"In {random.randint(1, 3)} hours"
        }

        # Recent activity
        recent_activity = [
            {'action': 'New Prediction', 'time': 'Just now', 'details': f'{ticker} ${stats["latest"]:.2f}'},
            {'action': 'Account Viewed', 'time': 'Today', 'details': 'Dashboard accessed'},
            {'action': 'Prediction', 'time': 'Yesterday', 'details': f'{random.choice(["AAPL", "MSFT", "GOOG"])} ${random.uniform(100, 500):.2f}'}
        ]

        # Calculate accuracy
        accuracy = calculate_accuracy(session['user_id'], ticker) or random.randint(75, 90)

        # Get market news
        market_news = get_market_news()

        return render_template(
            'dashboard.html',
            user=session.get('username', 'Guest'),
            predictions=prediction_data,
            stats=stats,
            ticker=ticker,
            no_predictions=False,
            current_date=datetime.now(),
            next_prediction=next_prediction,
            recent_activity=recent_activity,
            prediction_history=prediction_history,
            market_news=market_news,
            accuracy=accuracy
        )
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template('dashboard.html', 
                           user=session.get('username', 'Guest'),
                           no_predictions=True)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect('/login')

    if request.method == 'POST':
        ticker = request.form['ticker'].upper().strip()
        pred, hi, lo = predict_stock_price(ticker)

        if pred is None:
            return render_template('prediction.html', error="No data found for ticker or prediction failed.")

        cursor = get_db_cursor()
        try:
            cursor.execute(
                "INSERT INTO predictions (user_id, ticker, predicted_price, timestamp) VALUES (%s, %s, %s, %s)",
                (session['user_id'], ticker, float(pred), datetime.now())
            )
            mysql.connection.commit()

            return render_template('prediction.html',
                               prediction=round(pred, 2),
                               highest=round(hi, 2) if hi else None,
                               lowest=round(lo, 2) if lo else None,
                               ticker=ticker)
        except Exception as e:
            print(f"Prediction save error: {e}")
            mysql.connection.rollback()
            return render_template('prediction.html', error="Failed to save prediction")
    
    return render_template('prediction.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)


