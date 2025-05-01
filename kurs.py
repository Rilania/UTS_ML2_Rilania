import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

# Load model dan scaler
model = load_model('model_lstm.h5')
scaler = joblib.load('scaler.pkl')

# Parameter
n_lookback = 60

# UI
st.title("📈 Prediksi Kurs USD/IDR")
st.write("Model LSTM untuk memprediksi nilai tukar USD/IDR dalam beberapa hari ke depan.")

# Input jumlah hari ke depan
n_days = st.slider("Berapa hari ke depan yang ingin diprediksi?", 1, 180)

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV dengan data kurs (kolom: Close, index: Date)", type=["csv"])

if uploaded_file:
    # Baca dan parsing CSV (assumes Date as index)
    df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)

    # Tampilkan chart harga aktual
    st.subheader("Data Kurs Aktual:")
    st.line_chart(df['Close'])

    # Persiapan data
    last_close = df['Close'].values.reshape(-1, 1)
    last_scaled = scaler.transform(last_close)

    X_ = last_scaled[-n_lookback:]
    forecast_scaled = []

    # Prediksi
    for _ in range(n_days):
        X_input = X_.reshape(1, n_lookback, 1)
        y_pred = model.predict(X_input, verbose=0)
        forecast_scaled.append(y_pred[0, 0])
        X_ = np.vstack([X_[1:], [[y_pred[0, 0]]]])

    # Inverse transform hasil prediksi
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Buat tanggal prediksi
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.line_chart(forecast_df)

    # Download tombol
    st.download_button(
        "Download Forecast (CSV)",
        forecast_df.to_csv().encode('utf-8'),
        file_name="forecast.csv",
        mime="text/csv"
    )
