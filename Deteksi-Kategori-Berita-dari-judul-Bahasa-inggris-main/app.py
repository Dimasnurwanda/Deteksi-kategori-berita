import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk memuat model dan vektor
@st.cache_resource  # Caching untuk mempercepat load saat aplikasi dijalankan ulang
def load_model_and_vectorizer():
    with open("model/model_kategori_berita_pickle.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("model/vectorizer_kategori_berita_pickle.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Memuat model dan vectorizer
model, vectorizer = load_model_and_vectorizer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Hanya menyisakan huruf kecil dan spasi
    return text

# Fungsi untuk prediksi
def predict_category(input_text):
    cleaned_input = clean_text(input_text)
    input_tfidf = vectorizer.transform([cleaned_input])  # Mengubah teks menjadi vektor TF-IDF
    predicted_category = model.predict(input_tfidf)
    return predicted_category[0]

# Streamlit UI
st.title('Klasifikasi Kategori dari Judul Berita')
input_text = st.text_area("Masukkan Judul Berita")

# Tombol untuk melakukan analisis
if st.button('Analisis Judul Berita'):
    if input_text:
        predicted_category = predict_category(input_text)
        st.write(f"Kategori Berita adalah: **{predicted_category}**")
    else:
        st.write("Silakan masukkan Judul Berita.")
