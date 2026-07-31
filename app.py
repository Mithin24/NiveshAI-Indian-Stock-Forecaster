import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from transformers import pipeline
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- CONFIG ---
SEQ_LENGTH = 120
TICKERS = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS', 'ADANIPORTS.NS']
FEATURE_COLS = ['Close', 'Volume', 'RSI', 'Return', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low'] # Updated to 8 Features!

# --- Model Paths (Global) ---
MODELS_DIR = 'saved_models'
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'best_tuned_model.h5') # Updated model filename
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_TCS_new_features.pkl') # Updated scaler filename
META_MODEL_PATH = os.path.join(MODELS_DIR, 'meta_model.pkl')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.pkl') # Assuming you've saved the xgb_model from previous steps

# --- Global model variables are expected to be loaded by a previous cell ---
# The 'global' keyword within predict_next_day will correctly reference these.

# Initialize models as None, they will be loaded in the setup_models function
lstm_model = None
scaler = None
meta_model = None
xgb_model = None

def setup_models():
    global lstm_model, scaler, meta_model, xgb_model

    # Load the best tuned LSTM model
    try:
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        print("✅ Best tuned LSTM model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading LSTM model: {e}")

    # Load the scaler
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")

    # Load the Meta-Model
    try:
        with open(META_MODEL_PATH, 'rb') as f:
            meta_model = pickle.load(f)
        print("✅ Meta-model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading meta-model: {e}")

    # Load the XGBoost Model
    try:
        with open(XGB_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        print("✅ XGBoost model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading XGBoost model: {e}")

# Call setup_models to load them when the app starts
setup_models()

# NLP Pipeline
sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="tf")

def get_live_data(ticker):
    """Fetch and engineer features for live data"""
    df = yf.download(ticker, period="2y", progress=False)

    # Handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy() # Added 'Open' and 'High', 'Low' for candlestick
    df.dropna(inplace=True)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Return
    df['Return'] = df['Close'].pct_change()

    # Calculate MACD
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])

    # Calculate Bollinger Bands
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])

    df = df.dropna()
    return df

def create_lagged_features_live(data, lag=5, features_list=FEATURE_COLS):
    """Helper function to create lagged features for XGBoost prediction"""
    df_lagged = pd.DataFrame(index=data.index)
    for feature in features_list:
        for i in range(1, lag + 1):
            df_lagged[f'{feature}_lag_{i}'] = data[feature].shift(i)
    df_lagged['target_close'] = data['Close'].shift(-1) # Predict next day's Close price
    return df_lagged.dropna()

def predict_next_day(ticker, news):
    global lstm_model, scaler, meta_model, xgb_model # Access global variables

    # Check if models are loaded
    if lstm_model is None or scaler is None or meta_model is None or xgb_model is None:
        return "⚠️ Model files not loaded correctly. Please ensure models are saved and paths are correct.", None, None, None, None

    try:
        # 1. Get Live Data
        df = get_live_data(ticker)
        print(f"Live data shape: {df.shape}")

        if len(df) < SEQ_LENGTH + 5: # Need enough data for LSTM sequence and XGBoost lags
            return f"⚠️ Need more data. Have {len(df)} days, need at least {SEQ_LENGTH + 5} for both models.", None, None, None, None

        current_price = df['Close'].iloc[-1]

        # --- LSTM Prediction ---
        data_lstm = df[FEATURE_COLS].tail(SEQ_LENGTH).values.astype('float32')
        scaled_data_lstm = scaler.transform(data_lstm)
        X_input_lstm = scaled_data_lstm.reshape(1, SEQ_LENGTH, len(FEATURE_COLS))
        pred_scaled_lstm = lstm_model.predict(X_input_lstm)[0, 0]
        dummy_lstm = np.zeros((1, len(FEATURE_COLS)))
        dummy_lstm[0, 0] = pred_scaled_lstm
        lstm_pred_price = scaler.inverse_transform(dummy_lstm)[0, 0]

        # --- XGBoost Prediction ---
        # For XGBoost, we need the latest lagged features
        lag_period = 5 # This should match the lag used during XGBoost training
        df_for_xgb_lagged = create_lagged_features_live(df[FEATURE_COLS], lag=lag_period, features_list=FEATURE_COLS)

        if df_for_xgb_lagged.empty:
             return "⚠️ Not enough data to create lagged features for XGBoost prediction.", None, None, None, None

        XGB_TRAINING_FEATURES = xgb_model.feature_names_in_
        latest_xgb_features = df_for_xgb_lagged[XGB_TRAINING_FEATURES].tail(1)
        xgb_pred_price = xgb_model.predict(latest_xgb_features)[0]

        # --- Meta-Model Prediction ---
        new_meta_features = pd.DataFrame({
            'lstm_pred': [lstm_pred_price],
            'xgb_pred': [xgb_pred_price]
        })
        final_meta_prediction_raw = meta_model.predict(new_meta_features)[0]

        # 3. NLP Sentiment (apply to the final meta-model prediction)
        news_score = sentiment_pipe(news[:512])[0]
        star_map = {'1 star': -0.05, '2 stars': -0.025, '3 stars': 0.0, '4 stars': 0.025, '5 stars': 0.05}
        impact = star_map[news_score['label']]

        final_forecast_price = final_meta_prediction_raw * (1 + impact)

        # 4. Technicals
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        bollinger_high = df['Bollinger_High'].iloc[-1]
        bollinger_low = df['Bollinger_Low'].iloc[-1]

        # Generate Plots
        # Closing Price Plot
        fig_close = plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')
        plt.title(f'{ticker} Closing Price Trend')
        plt.xlabel('Date')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # MACD Plot
        fig_macd = plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['MACD'], label='MACD', color='red')
        plt.plot(df.index, df['MACD_Signal'], label='MACD Signal', color='green')
        plt.title(f'{ticker} MACD Trend')
        plt.xlabel('Date')
        plt.ylabel('MACD Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Bollinger Bands Plot
        fig_bb = plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')
        plt.plot(df.index, df['Bollinger_High'], label='Bollinger High', color='orange', linestyle='--')
        plt.plot(df.index, df['Bollinger_Low'], label='Bollinger Low', color='purple', linestyle='--')
        plt.title(f'{ticker} Bollinger Bands Trend')
        plt.xlabel('Date')
        plt.ylabel('Price (INR)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Candlestick Chart (Plotly)
        fig_candlestick = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        fig_candlestick.update_layout(title_text=f'{ticker} Candlestick Chart', xaxis_rangeslider_visible=False)

        output_markdown = f"""
        **🇮🇳 Stock:** {ticker}
        **Current Price:** ₹{current_price:.2f}

        **📊 Technicals:**
        - RSI (14): {rsi:.2f}
        - MACD: {macd:.2f}
        - MACD Signal: {macd:.2f}
        - Bollinger High: {bollinger_high:.2f}
        - Bollinger Low: {bollinger_low:.2f}

        **🧠 LSTM Base Prediction:** ₹{lstm_pred_price:.2f}
        **🌳 XGBoost Base Prediction:** ₹{xgb_pred_price:.2f}
        **융 Meta-Model Prediction (before sentiment):** ₹{final_meta_prediction_raw:.2f}

        **📰 News Sentiment:** {news_score['label']} ({news_score['score']:.2%})

        **🎯 Final Forecast (with sentiment adjustment):** ₹{final_forecast_price:.2f}
        """

        return output_markdown, fig_close, fig_macd, fig_bb, fig_candlestick

    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

# --- GRADIO UI ---
demo = gr.Interface(
    fn=predict_next_day,
    inputs=[
        gr.Dropdown(TICKERS, label="Select Indian Stock", value="TCS.NS"),
        gr.Textbox(label="Recent News Headline", placeholder="e.g. TCS wins new deal...")
    ],
    outputs=[
        gr.Markdown(label="Prediction and Technicals"),
        gr.Plot(label="Closing Price Trend"),
        gr.Plot(label="MACD Trend"),
        gr.Plot(label="Bollinger Bands Trend"),
        gr.Plot(label="Candlestick Chart") # Added for candlestick chart
    ],
    title="🇮🇳 Indian Stock LSTM Forecaster",
    description="Deep Learning + NLP Stock Forecaster"
)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",   # required so Render can reach it
        server_port=port,
        share=False
    )
# To deploy to Hugging Face Spaces, you typically don't run demo.launch() in the app.py file itself.
# The Gradio Spaces environment handles the launching.
# For local testing, you can uncomment the line below:
# demo.launch()
