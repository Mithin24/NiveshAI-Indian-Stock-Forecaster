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
import traceback

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

    print("Loading models...")

    try:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(LSTM_MODEL_PATH)

        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(SCALER_PATH)

        if not os.path.exists(META_MODEL_PATH):
            raise FileNotFoundError(META_MODEL_PATH)

        if not os.path.exists(XGB_MODEL_PATH):
            raise FileNotFoundError(XGB_MODEL_PATH)

        lstm_model = tf.keras.models.load_model(
            LSTM_MODEL_PATH,
            compile=False
        )

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        with open(META_MODEL_PATH, "rb") as f:
            meta_model = pickle.load(f)

        with open(XGB_MODEL_PATH, "rb") as f:
            xgb_model = pickle.load(f)

        print("✅ All models loaded successfully!")

    except Exception:
        print("❌ Model loading failed")
        traceback.print_exc()

sentiment_pipe = None

def get_sentiment_pipeline():
    global sentiment_pipe

    if sentiment_pipe is None:
        print("Loading Hugging Face sentiment model...")

        from transformers import pipeline

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

        if lstm_model is None:
            return "❌ Models not loaded."

        df = get_live_data(ticker)

        data_lstm = df[FEATURE_COLS].tail(SEQ_LENGTH).values.astype(np.float32)

        scaled = scaler.transform(data_lstm)

        scaled = scaled.reshape(
            1,
            SEQ_LENGTH,
            len(FEATURE_COLS)
        )

        lstm_scaled = lstm_model.predict(
            scaled,
            verbose=0
        )[0][0]

        dummy = np.zeros((1, len(FEATURE_COLS)))

        dummy[0, 0] = lstm_scaled

        lstm_prediction = scaler.inverse_transform(dummy)[0, 0]

        latest = pd.DataFrame(
            [df[FEATURE_COLS].iloc[-1]],
            columns=xgb_model.feature_names_in_
        )

        xgb_prediction = xgb_model.predict(latest)[0]

        meta_prediction = meta_model.predict(
            pd.DataFrame({
                "lstm_pred": [lstm_prediction],
                "xgb_pred": [xgb_prediction]
            })
        )[0]

        pipe = get_sentiment_pipeline()

        sentiment = pipe(news[:512])[0]

        impact = {
            "1 star": -0.05,
            "2 stars": -0.025,
            "3 stars": 0.0,
            "4 stars": 0.025,
            "5 stars": 0.05,
            "NEGATIVE": -0.05,
            "POSITIVE": 0.05
        }

        adjustment = impact.get(
            sentiment["label"],
            0.0
        )

        final_prediction = meta_prediction * (1 + adjustment)

        return (
            f"Predicted Price: ₹{final_prediction:.2f}\n"
            f"Sentiment: {sentiment['label']}"
        )

    except Exception as e:
        traceback.print_exc()
        return f"Prediction Error:\n{e}"

print("Creating Gradio interface...")

demo = gr.Interface(
    fn=predict_next_day,
    inputs=[
        gr.Dropdown(
            choices=TICKERS,
            label="Stock"
        ),
        gr.Textbox(
            lines=6,
            label="News"
        )
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="NiveshAI - Indian Stock Forecaster",
    description="LSTM + XGBoost + Meta Model + Sentiment Analysis"
)

if __name__ == "__main__":

    print("Launching Gradio...")

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
