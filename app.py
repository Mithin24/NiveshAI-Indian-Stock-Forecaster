import gradio as gr
import yfinance as yf
import os
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from transformers import pipeline
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- CONFIG ---
SEQ_LENGTH = 120
TICKERS = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS', 'ADANIPORTS.NS']
FEATURE_COLS = ['Close', 'Volume', 'RSI', 'Return', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

# --- Model Paths ---
MODELS_DIR = 'saved_models'
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'best_tuned_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_TCS_new_features.pkl')
META_MODEL_PATH = os.path.join(MODELS_DIR, 'meta_model.pkl')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.pkl')

lstm_model = None
scaler = None
meta_model = None
xgb_model = None

def setup_models():
    global lstm_model, scaler, meta_model, xgb_model
    try:
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        with open(META_MODEL_PATH, 'rb') as f: meta_model = pickle.load(f)
        with open(XGB_MODEL_PATH, 'rb') as f: xgb_model = pickle.load(f)
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

setup_models()
sentiment_pipe = None

def get_sentiment_pipeline():
    global sentiment_pipe

    if sentiment_pipe is None:
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="tf"
        )

    return sentiment_pipe
def get_live_data(ticker):
    df = yf.download(
        ticker,
        period="5y",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Technical Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bb.bollinger_hband()
    df['Bollinger_Low'] = bb.bollinger_lband()

    df['Return'] = df['Close'].pct_change()

    df = df.dropna()

    if len(df) < SEQ_LENGTH:
        raise ValueError(
            f"Only {len(df)} rows available after preprocessing."
        )

    return df

def predict_next_day(ticker, news):
    try:
        df = get_live_data(ticker)

        if df.empty:
            return "❌ No market data."

        if len(df) < SEQ_LENGTH:
            return f"❌ Need at least {SEQ_LENGTH} rows, got {len(df)}"

        # ---------- LSTM ----------
        data_lstm = df[FEATURE_COLS].tail(SEQ_LENGTH).values.astype("float32")

        scaled_lstm = scaler.transform(data_lstm)
        scaled_lstm = scaled_lstm.reshape(
            1,
            SEQ_LENGTH,
            len(FEATURE_COLS)
        )

        pred_lstm_scaled = lstm_model.predict(scaled_lstm, verbose=0)[0, 0]

        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0, 0] = pred_lstm_scaled
        lstm_p = scaler.inverse_transform(dummy)[0, 0]

        # ---------- XGB ----------
        lag = 5
        xgb_features = {}

        for feature in FEATURE_COLS:
            for i in range(1, lag + 1):
                xgb_features[f"{feature}_lag_{i}"] = df[feature].iloc[-i]

        xgb_input = pd.DataFrame([xgb_features])
        xgb_input = xgb_input[xgb_model.feature_names_in_]

        xgb_p = xgb_model.predict(xgb_input)[0]

        # ---------- Meta ----------
        meta_p = meta_model.predict(
            pd.DataFrame({
                "lstm_pred": [lstm_p],
                "xgb_pred": [xgb_p]
            })
        )[0]

        # ---------- Sentiment ----------
        impact = 0
        sentiment = "NEUTRAL"

        if news.strip():
            try:
                res = get_sentiment_pipeline()(news[:512])[0]
                sentiment = res["label"]

                if sentiment == "POSITIVE":
                    impact = 0.03
                elif sentiment == "NEGATIVE":
                    impact = -0.03

            except Exception as e:
                print("Sentiment Error:", e)

        final_price = meta_p * (1 + impact)

        return (
            f"Predicted Price: ₹{final_price:.2f}\n\n"
            f"LSTM: ₹{lstm_p:.2f}\n"
            f"XGBoost: ₹{xgb_p:.2f}\n"
            f"Meta: ₹{meta_p:.2f}\n"
            f"Sentiment: {sentiment}"
        )

    except Exception as e:
        return f"❌ {e}"

demo = gr.Interface(fn=predict_next_day, inputs=[gr.Dropdown(TICKERS), gr.Textbox()], outputs="text")
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
