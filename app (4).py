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
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return sentiment_pipe
def get_live_data(ticker):
    df = yf.download(ticker, period="2y", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['Return'] = df['Close'].pct_change()
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
    return df.dropna()

def predict_next_day(ticker, news):
    try:
        df = get_live_data(ticker)
        # LSTM part
        data_lstm = df[FEATURE_COLS].tail(SEQ_LENGTH).values.astype('float32')
        scaled_lstm = scaler.transform(data_lstm).reshape(1, SEQ_LENGTH, len(FEATURE_COLS))
        pred_lstm_scaled = lstm_model.predict(scaled_lstm)[0,0]
        dummy = np.zeros((1, len(FEATURE_COLS)))
        dummy[0,0] = pred_lstm_scaled
        lstm_p = scaler.inverse_transform(dummy)[0,0]
        # XGB part (Simplified for Space)
        xgb_p = xgb_model.predict(pd.DataFrame([df[FEATURE_COLS].iloc[-1].values], columns=xgb_model.feature_names_in_))[0]
        # Meta & Sentiment
        meta_p = meta_model.predict(pd.DataFrame({'lstm_pred': [lstm_p], 'xgb_pred': [xgb_p]}))[0]
        res = get_sentiment_pipeline()(news[:512])[0]
        impact = {'1 star':-0.05,'2 stars':-0.025,'3 stars':0.0,'4 stars':0.025,'5 stars':0.05}[res['label']]
        final_p = meta_p * (1 + impact)
        return f"Predicted Price: ₹{final_p:.2f} (Sentiment: {res['label']})"
    except Exception as e:
        return str(e)

demo = gr.Interface(fn=predict_next_day, inputs=[gr.Dropdown(TICKERS), gr.Textbox()], outputs="text")
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000))
    )
