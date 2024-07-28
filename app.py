import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

# Memuat model
model = tf.keras.models.load_model('sentiment_analysis_model.h5')

# Memuat data yang sudah diproses
processed_data = pd.read_csv('processed_data.csv')

# Fungsi untuk prediksi
def predict_sentiment(text):
    # Preprocess text input sesuai kebutuhan
    # Misalnya, jika data Anda memerlukan tokenisasi atau normalisasi
    # text_preprocessed = ...
    # pred = model.predict(np.array([text_preprocessed]))
    # return pred[0][0]
    pass

# Membuat antarmuka Streamlit
st.title('Sentiment Analysis App')
user_input = st.text_area("Enter text:")
   
if st.button("Predict"):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(f"Sentiment score: {result}")
    else:
        st.write("Please enter some text.")

st.write("Processed Data Preview:")
st.dataframe(processed_data.head())
