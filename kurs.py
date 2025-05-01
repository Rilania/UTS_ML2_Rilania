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
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Tangani tanggal: cek apakah ada kolom 'Date'
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        # Kalau tidak ada, cek apakah index sudah datetime
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            st.error("❌ Kolom 'Date' tidak ditemukan dan index bukan datetime. Harap periksa file.")
            st.stop()

    # Cek kolom 'Close'
    if 'Close' not in df.columns:
        st.error("❌ Kolom 'Close' tidak ditemukan. Pastikan file CSV berisi kolom 'Close'.")
        st.stop()

    # Tampilkan data aktual
    st.subheader("📊 Data Kurs Aktual:")
    st.line_chart(df['Close'])

    # Proses scaling
    last_close = df['Close'].values.reshape(-1, 1)
    try:
        last_scaled = scaler.transform(last_close)
    except Exception as e:
        st.error(f"❌ Error saat menormalisasi data dengan scaler: {e}")
        st.stop()

    # Inisialisasi data untuk prediksi
    X_ = last_scaled[-n_lookback:]
    forecast_scaled = []

    # Prediksi
    for _ in range(n_days):
        X_input = X_.reshape(1, n_lookback, 1)
        y_pred = model.predict(X_input, verbose=0)
        forecast_scaled.append(y_pred[0, 0])
        X_ = np.vstack([X_[1:], [[y_pred[0, 0]]]])

    # Invers scaling hasil prediksi
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Buat tanggal hasil prediksi
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

    # Tampilkan hasil prediksi
    st.subheader("📈 Hasil Prediksi:")
    st.line_chart(forecast_df)

    # Download button
    st.download_button(
        "📥 Download Hasil Prediksi (CSV)",
        forecast_df.to_csv().encode('utf-8'),
        "forecast.csv",
        "text/csv"
    )
