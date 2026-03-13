"""
Stock Price Predictor — Streamlit Web Application
Udacity Data Scientist Nanodegree Capstone

Run with:
    pip install streamlit yfinance scikit-learn xgboost pandas numpy matplotlib
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; color: #1F497D; }
    .subtitle   { font-size: 1.1rem; color: #666; margin-bottom: 1.5rem; }
    .metric-box { background: #f0f4f8; border-radius: 10px; padding: 1rem; text-align: center; }
    .stAlert    { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Feature engineering ───────────────────────────────────────────────────────
FEATURE_COLS = [
    'lag_1','lag_2','lag_3','lag_5','lag_10',
    'sma_5','sma_10','sma_20','sma_50',
    'ema_5','ema_10','ema_20',
    'price_to_sma20','price_to_sma50','sma5_to_sma20',
    'return_1d','return_5d','return_10d','return_20d',
    'vol_5d','vol_10d','vol_20d',
    'vol_ratio','log_volume',
    'rsi_14','bb_position','bb_width',
    'macd','macd_signal','macd_hist',
    'hl_range','hl_range_ma5',
    'day_of_week','month','quarter'
]

def add_features(df):
    df = df.copy().sort_index()
    close = df['Close']
    for lag in [1, 2, 3, 5, 10]:
        df[f'lag_{lag}'] = close.shift(lag)
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = close.rolling(window).mean()
        df[f'ema_{window}'] = close.ewm(span=window).mean()
    df['price_to_sma20'] = close / df['sma_20']
    df['price_to_sma50'] = close / df['sma_50']
    df['sma5_to_sma20']  = df['sma_5'] / df['sma_20']
    df['return_1d']  = close.pct_change(1)
    df['return_5d']  = close.pct_change(5)
    df['return_10d'] = close.pct_change(10)
    df['return_20d'] = close.pct_change(20)
    for window in [5, 10, 20]:
        df[f'vol_{window}d'] = df['return_1d'].rolling(window).std()
    df['vol_avg5']   = df['Volume'].rolling(5).mean()
    df['vol_ratio']  = df['Volume'] / (df['vol_avg5'] + 1)
    df['log_volume'] = np.log1p(df['Volume'])
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    bb_std = close.rolling(20).std()
    df['bb_upper']    = df['sma_20'] + 2 * bb_std
    df['bb_lower']    = df['sma_20'] - 2 * bb_std
    df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / (df['sma_20'] + 1e-9)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['hl_range']     = (df['High'] - df['Low']) / (close + 1e-9)
    df['hl_range_ma5'] = df['hl_range'].rolling(5).mean()
    df['day_of_week'] = df.index.dayofweek
    df['month']       = df.index.month
    df['quarter']     = df.index.quarter
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, start, end):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error downloading {ticker}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def train_model(ticker, train_start, train_end, horizon, model_type):
    df = load_data(ticker, train_start, (pd.Timestamp(train_end) + timedelta(days=5)).strftime('%Y-%m-%d'))
    if df is None or len(df) < 60:
        return None, None
    df = add_features(df)
    df['target'] = df['Close'].shift(-horizon)
    df.dropna(inplace=True)
    train = df[df.index <= train_end]
    X, y = train[FEATURE_COLS], train['target']
    if len(X) == 0:
        return None, None
    if model_type == 'XGBoost' and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    elif model_type == 'XGBoost':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, df

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Stockmarket_graph.svg/320px-Stockmarket_graph.svg.png", use_column_width=True)
    st.markdown("## ⚙️ Configuration")

    ticker = st.selectbox("Stock Ticker", ["AAPL", "GOOG", "AMZN", "MSFT", "NVDA"], index=0)
    model_type = st.radio("Model", ["Random Forest", "XGBoost"])
    horizon = st.select_slider("Prediction Horizon (days)", options=[1, 7, 14, 28], value=7)

    st.markdown("### Training Period")
    train_start = st.date_input("Start", value=datetime(2019, 1, 1))
    train_end   = st.date_input("End",   value=datetime(2023, 12, 31))

    st.markdown("### Test Period")
    test_start = st.date_input("Test Start", value=datetime(2024, 1, 1))
    test_end   = st.date_input("Test End",   value=datetime(2024, 12, 31))

    run_btn = st.button("Train & Predict", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Built with Random Forest & XGBoost | Udacity Capstone")

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Forecast AAPL · GOOG · AMZN using ensemble machine learning</div>', unsafe_allow_html=True)

if not run_btn:
    st.info("Configure your settings in the sidebar and click **Train & Predict** to get started.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Random Forest\nEnsemble of 200 decision trees. Robust to noise, great interpretability via feature importance.")
    with col2:
        st.markdown("### XGBoost\nGradient-boosted trees with L1/L2 regularization. Generally higher accuracy than RF at longer horizons.")
    with col3:
        st.markdown("### Features\n34 technical indicators: RSI, MACD, Bollinger Bands, lag prices, moving averages, volume signals.")
    st.stop()

# ── Training ──────────────────────────────────────────────────────────────────
with st.spinner(f"Training {model_type} on {ticker} ({horizon}-day horizon)..."):
    model, full_df = train_model(
        ticker,
        train_start.strftime('%Y-%m-%d'),
        train_end.strftime('%Y-%m-%d'),
        horizon,
        model_type
    )

if model is None:
    st.error("Training failed. Please check your date range and ensure yfinance is installed.")
    st.stop()

# ── Evaluation on test set ────────────────────────────────────────────────────
test_df = load_data(ticker,
    (pd.Timestamp(test_start) - timedelta(days=100)).strftime('%Y-%m-%d'),
    test_end.strftime('%Y-%m-%d'))

if test_df is None or len(test_df) < 30:
    st.error("Could not load test data.")
    st.stop()

test_df = add_features(test_df)
test_df['target'] = test_df['Close'].shift(-horizon)
test_df.dropna(inplace=True)
test_filtered = test_df[test_df.index >= pd.Timestamp(test_start)]

if len(test_filtered) == 0:
    st.warning("No test data found in the specified test period.")
    st.stop()

X_test  = test_filtered[FEATURE_COLS]
y_test  = test_filtered['target']
y_pred  = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# ── Metrics row ───────────────────────────────────────────────────────────────
st.success(f"Model trained successfully on {ticker} | {model_type} | {horizon}-day horizon")
st.markdown("### Model Performance on Test Set")

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"${rmse:.2f}")
col2.metric("MAE",  f"${mae:.2f}")
col3.metric("R²",   f"{r2:.4f}")
col4.metric("MAPE", f"{mape:.2f}%", delta=f"{'Within 5%' if mape <= 5 else '>5%'}", delta_color="normal")

# ── Prediction chart ──────────────────────────────────────────────────────────
st.markdown("### Predicted vs Actual Price")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_test.index, y_test.values,  label='Actual',    color='#1F497D', linewidth=2)
ax.plot(y_test.index, y_pred,         label='Predicted', color='#FF5722', linewidth=1.8, linestyle='--')
ax.fill_between(y_test.index,
                y_pred * 0.97, y_pred * 1.03,
                alpha=0.15, color='#FF5722', label='±3% band')
ax.set_title(f'{ticker} — {horizon}-Day Ahead Predictions | {model_type}', fontsize=13, fontweight='bold')
ax.set_ylabel('Adj. Close (USD)')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
st.pyplot(fig)

# ── Feature importance ────────────────────────────────────────────────────────
st.markdown("### Top 15 Feature Importances")

if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).nlargest(15).sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    importances.plot(kind='barh', ax=ax2, color='#2196F3', alpha=0.8)
    ax2.set_title(f'Feature Importance — {model_type}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig2)

# ── Query interface ───────────────────────────────────────────────────────────
st.markdown("### Query: Predict a Specific Date")
query_date = st.date_input("Select a future date to predict", value=datetime(2025, 1, 15))

if st.button("Predict Price"):
    q_start = (pd.Timestamp(query_date) - timedelta(days=100)).strftime('%Y-%m-%d')
    q_end   = query_date.strftime('%Y-%m-%d')
    q_df = load_data(ticker, q_start, q_end)
    if q_df is not None and len(q_df) >= 50:
        q_df = add_features(q_df)
        q_df.dropna(inplace=True)
        if len(q_df) > 0:
            last_row = q_df.iloc[[-1]][FEATURE_COLS]
            pred_price = model.predict(last_row)[0]
            last_actual = q_df['Close'].iloc[-1]
            pct_change  = (pred_price - last_actual) / last_actual * 100
            st.success(f"**{ticker} Predicted Adj. Close in ~{horizon} days from {q_df.index[-1].date()}:** "
                       f"${pred_price:.2f}  ({pct_change:+.2f}% vs last close of ${last_actual:.2f})")
        else:
            st.warning("Not enough data to compute features.")
    else:
        st.warning("Could not retrieve enough data for this date. Make sure yfinance is installed and the date is valid.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Udacity Data Scientist Nanodegree Capstone | Investment & Trading | Random Forest + XGBoost by Mariana Morao")
