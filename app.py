import os
# Force TensorFlow to CPU and quiet its logs (saves a bit of RAM + noise)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import ta
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – important for servers
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- CONFIG ---
SEQ_LENGTH = 120
TICKERS = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'SBIN.NS', 'ADANIPORTS.NS']
FEATURE_COLS = ['Close', 'Volume', 'RSI', 'Return', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']

MODELS_DIR = 'saved_models'
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, 'best_tuned_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler_TCS_new_features.pkl')
META_MODEL_PATH = os.path.join(MODELS_DIR, 'meta_model.pkl')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.pkl')

# Models start as None – loaded only on first prediction (lazy)
lstm_model = None
scaler = None
meta_model = None
xgb_model = None
_models_loaded = False


def load_models_once():
    """Load all models the first time they are needed."""
    global lstm_model, scaler, meta_model, xgb_model, _models_loaded
    if _models_loaded:
        return True

    try:
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(META_MODEL_PATH, 'rb') as f:
            meta_model = pickle.load(f)
        with open(XGB_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        _models_loaded = True
        print("✅ All models loaded successfully (lazy)")
        return True
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return False


def simple_sentiment(text: str) -> float:
    """Tiny keyword-based sentiment → impact factor. Zero extra memory."""
    text = text.lower()
    positive = ["win", "wins", "deal", "growth", "profit", "rise", "up", "boost", "strong", "beat", "positive"]
    negative = ["loss", "fall", "down", "miss", "weak", "cut", "decline", "negative", "lawsuit", "probe"]

    score = 0
    for w in positive:
        if w in text:
            score += 1
    for w in negative:
        if w in text:
            score -= 1

    # Map to a small price impact
    if score >= 2:
        return 0.03
    if score == 1:
        return 0.015
    if score == -1:
        return -0.015
    if score <= -2:
        return -0.03
    return 0.0


def get_live_data(ticker):
    df = yf.download(ticker, period="2y", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.dropna(inplace=True)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Return
    df['Return'] = df['Close'].pct_change()

    # MACD & Bollinger
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])

    df = df.dropna()
    return df


def create_lagged_features_live(data, lag=5, features_list=FEATURE_COLS):
    df_lagged = pd.DataFrame(index=data.index)
    for feature in features_list:
        for i in range(1, lag + 1):
            df_lagged[f'{feature}_lag_{i}'] = data[feature].shift(i)
    df_lagged['target_close'] = data['Close'].shift(-1)
    return df_lagged.dropna()


def predict_next_day(ticker, news):
    if not load_models_once():
        return "⚠️ Model files not loaded. Check that saved_models/ exists on the server.", None, None, None, None

    try:
        df = get_live_data(ticker)

        if len(df) < SEQ_LENGTH + 5:
            return (f"⚠️ Need more data. Have {len(df)} days, need ≥ {SEQ_LENGTH + 5}.",
                    None, None, None, None)

        current_price = float(df['Close'].iloc[-1])

        # --- LSTM ---
        data_lstm = df[FEATURE_COLS].tail(SEQ_LENGTH).values.astype('float32')
        scaled_data_lstm = scaler.transform(data_lstm)
        X_input_lstm = scaled_data_lstm.reshape(1, SEQ_LENGTH, len(FEATURE_COLS))
        pred_scaled_lstm = lstm_model.predict(X_input_lstm, verbose=0)[0, 0]
        dummy_lstm = np.zeros((1, len(FEATURE_COLS)))
        dummy_lstm[0, 0] = pred_scaled_lstm
        lstm_pred_price = float(scaler.inverse_transform(dummy_lstm)[0, 0])

        # --- XGBoost ---
        lag_period = 5
        df_for_xgb_lagged = create_lagged_features_live(df[FEATURE_COLS], lag=lag_period)
        if df_for_xgb_lagged.empty:
            return "⚠️ Not enough data for XGBoost lagged features.", None, None, None, None

        XGB_TRAINING_FEATURES = xgb_model.feature_names_in_
        latest_xgb_features = df_for_xgb_lagged[XGB_TRAINING_FEATURES].tail(1)
        xgb_pred_price = float(xgb_model.predict(latest_xgb_features)[0])

        # --- Meta model ---
        new_meta_features = pd.DataFrame({
            'lstm_pred': [lstm_pred_price],
            'xgb_pred': [xgb_pred_price]
        })
        final_meta_prediction_raw = float(meta_model.predict(new_meta_features)[0])

        # --- Simple sentiment ---
        impact = simple_sentiment(news or "")
        final_forecast_price = final_meta_prediction_raw * (1 + impact)

        # Technicals
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        macd_signal = float(df['MACD_Signal'].iloc[-1])
        bollinger_high = float(df['Bollinger_High'].iloc[-1])
        bollinger_low = float(df['Bollinger_Low'].iloc[-1])

        # ---- Plots (closed after creation to free RAM) ----
        fig_close = plt.figure(figsize=(10, 4))
        plt.plot(df.index, df['Close'], label='Close', color='blue')
        plt.title(f'{ticker} Closing Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig_macd = plt.figure(figsize=(10, 4))
        plt.plot(df.index, df['MACD'], label='MACD', color='red')
        plt.plot(df.index, df['MACD_Signal'], label='Signal', color='green')
        plt.title(f'{ticker} MACD')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig_bb = plt.figure(figsize=(10, 4))
        plt.plot(df.index, df['Close'], label='Close', color='blue')
        plt.plot(df.index, df['Bollinger_High'], label='BB High', color='orange', linestyle='--')
        plt.plot(df.index, df['Bollinger_Low'], label='BB Low', color='purple', linestyle='--')
        plt.title(f'{ticker} Bollinger Bands')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig_candlestick = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close']
        )])
        fig_candlestick.update_layout(
            title=f'{ticker} Candlestick',
            xaxis_rangeslider_visible=False,
            height=400
        )

        sentiment_label = "Positive" if impact > 0 else "Negative" if impact < 0 else "Neutral"

        output_markdown = f"""
**🇮🇳 Stock:** {ticker}  
**Current Price:** ₹{current_price:.2f}

**📊 Technicals:**
- RSI (14): {rsi:.2f}
- MACD: {macd:.2f}
- MACD Signal: {macd_signal:.2f}
- Bollinger High: {bollinger_high:.2f}
- Bollinger Low: {bollinger_low:.2f}

**🧠 LSTM Prediction:** ₹{lstm_pred_price:.2f}  
**🌳 XGBoost Prediction:** ₹{xgb_pred_price:.2f}  
**🔗 Meta-Model (before sentiment):** ₹{final_meta_prediction_raw:.2f}

**📰 News Sentiment:** {sentiment_label} (impact {impact:+.1%})

**🎯 Final Forecast:** ₹{final_forecast_price:.2f}
"""

        # Explicitly close matplotlib figures later if needed
        # (Gradio will handle the returned objects)

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
        gr.Plot(label="Candlestick Chart")
    ],
    title="🇮🇳 Indian Stock LSTM Forecaster",
    description="Deep Learning + NLP Stock Forecaster"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
