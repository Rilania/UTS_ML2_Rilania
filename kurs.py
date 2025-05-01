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

    # Cek kolom 'Close'
    if 'Close' not in df.columns:
        st.error("❌ Kolom 'Close' tidak ditemukan. Pastikan file CSV berisi kolom 'Close'.")
    else:
        # Tangani tanggal
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            # Coba gunakan index jika sudah datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                st.warning("Index tidak bertipe datetime dan kolom 'Date' tidak ditemukan. Harap periksa file.")
                st.stop()

        # Tampilkan data aktual
        st.subheader("📊 Data Kurs Aktual:")
        st.line_chart(df['Close'])

        # Prediksi
        last_close = df['Close'].values.reshape(-1, 1)
        last_scaled = scaler.transform(last_close)

        X_ = last_scaled[-n_lookback:]
        forecast_scaled = []

        for _ in range(n_days):
            X_input = X_.reshape(1, n_lookback, 1)
            y_pred = model.predict(X_input, verbose=0)
            forecast_scaled.append(y_pred[0, 0])
            X_ = np.vstack([X_[1:], [[y_pred[0, 0]]]])

        forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
        forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

        # Tampilkan hasil prediksi
        st.subheader("📈 Hasil Prediksi:")
        st.line_chart(forecast_df)

        # Tombol download
        st.download_button(
            "📥 Download Hasil Prediksi (CSV)",
            forecast_df.to_csv().encode('utf-8'),
            "forecast.csv",
            "text/csv"
        )
