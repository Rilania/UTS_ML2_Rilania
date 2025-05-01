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

# Input hari
n_days = st.slider("Berapa hari ke depan yang ingin diprediksi?", 1, 180)

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])

if uploaded_file:
    # Baca data dan bersihkan
    df = pd.read_csv(uploaded_file, skiprows=2)  # Lewati header Yahoo
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)
    df = df[['Close']]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(inplace=True)

    st.subheader("Data Kurs Aktual:")
    st.line_chart(df['Close'])

    # Preprocessing input
    last_close = df['Close'].values.reshape(-1, 1)
    last_scaled = scaler.transform(last_close)
    X_ = last_scaled[-n_lookback:]

    # Forecast loop
    forecast_scaled = []
    for _ in range(n_days):
        X_input = X_.reshape(1, n_lookback, 1)
        y_pred = model.predict(X_input, verbose=0)
        forecast_scaled.append(y_pred[0, 0])
        X_ = np.vstack([X_[1:], [[y_pred[0, 0]]]])

    # Inverse transform
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Buat tanggal untuk forecast
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

    st.subheader("Hasil Prediksi:")
    st.line_chart(forecast_df)

    # Tombol unduh
    st.download_button(
        label="📥 Download Forecast (CSV)",
        data=forecast_df.to_csv().encode('utf-8'),
        file_name='forecast.csv',
        mime='text/csv'
    )
