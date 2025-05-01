import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

# Load model dan scaler
model = load_model('model_lstm.h5')
scaler = joblib.load('scaler.pkl')

# Konfigurasi
n_lookback = 60

# Judul
st.title("📈 Prediksi Kurs USD/IDR")
st.write("Model LSTM untuk memprediksi nilai tukar USD/IDR dalam beberapa hari ke depan.")

# Input hari prediksi
n_days = st.slider("Berapa hari ke depan yang ingin diprediksi?", 1, 180)

# Upload data
uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)

    # Coba ubah index ke datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        st.error("Index tidak bertipe datetime dan kolom 'Date' tidak ditemukan. Harap periksa file.")
        st.stop()

    # Validasi kolom Close
    if 'Close' not in df.columns:
        st.error("Kolom 'Close' tidak ditemukan di dataset.")
        st.stop()

    if df['Close'].isnull().any():
        st.error("Dataset mengandung nilai kosong (NaN) di kolom 'Close'. Harap bersihkan datanya.")
        st.stop()

    if df.shape[0] < n_lookback:
        st.error(f"Dataset minimal harus punya {n_lookback} baris.")
        st.stop()

    st.subheader("📊 Data Kurs Aktual:")
    st.line_chart(df['Close'])

    # Preprocess
    last_close = df['Close'].values.reshape(-1, 1)
    last_scaled = scaler.transform(last_close)
    X_ = last_scaled[-n_lookback:]
    forecast_scaled = []

    for _ in range(n_days):
        X_input = X_.reshape(1, n_lookback, 1)
        y_pred = model.predict(X_input, verbose=0)
        forecast_scaled.append(y_pred[0, 0])
        X_ = np.vstack([X_[1:], [[y_pred[0, 0]]]])

    # Inverse transform
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

    # Tampilkan hasil
    st.subheader("📉 Hasil Prediksi:")
    st.line_chart(forecast_df)

    # Download
    st.download_button("⬇️ Download Forecast (CSV)", forecast_df.to_csv().encode('utf-8'), "forecast.csv", "text/csv")
