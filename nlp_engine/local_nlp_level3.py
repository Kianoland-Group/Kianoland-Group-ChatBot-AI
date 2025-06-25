import os
import json
import random
import joblib
import pandas as pd
import numpy as np

# NLTK untuk tokenisasi
import nltk

# Scikit-learn untuk membangun model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report

# Sentence Transformers untuk membuat embeddings
from sentence_transformers import SentenceTransformer

# --- BAGIAN 1: PEMROSESAN DATA ---

def load_and_prepare_data(intents_dir='../dialogflow_kianoland/intents/'):
    """
    Memuat semua file intent, mengekstrak frasa pelatihan,
    dan menyiapkannya dalam format DataFrame pandas.
    """
    data = []
    responses = {}
    
    # Memuat setiap file JSON dari direktori intents
    for filename in os.listdir(intents_dir):
        if filename.endswith('.json') and not filename.startswith('usersays'):
            intent_name = os.path.splitext(filename)[0]
            
            with open(os.path.join(intents_dir, filename), 'r', encoding='utf-8') as f:
                intent_data = json.load(f)

                # Menyimpan respons untuk setiap intent
                if 'messages' in intent_data and len(intent_data['messages']) > 0:
                    # Pastikan struktur message sesuai
                    messages_list = intent_data['messages']
                    valid_responses = [msg.get('text', {}).get('text', [None])[0] for msg in messages_list]
                    responses[intent_name] = [r for r in valid_responses if r]


                # Menyimpan frasa pelatihan
                if 'trainingPhrases' in intent_data:
                    for phrase in intent_data['trainingPhrases']:
                        # Menggabungkan bagian-bagian teks menjadi satu kalimat
                        text = "".join([part['text'] for part in phrase['parts']])
                        data.append({'training_phrase': text.strip(), 'intent': intent_name})
                        
    # Simpan responses ke file JSON terpisah untuk digunakan saat prediksi
    with open('responses.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    return pd.DataFrame(data)

# --- BAGIAN 2: PEMBUATAN FITUR CUSTOM (LEVEL 2) ---

class SentenceTransformerEmbeddings(BaseEstimator, TransformerMixin):
    """
    Sebuah transformer custom untuk scikit-learn pipeline yang mengubah
    teks menjadi vector embeddings menggunakan SentenceTransformer.
    """
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # Model ini bagus untuk berbagai bahasa, termasuk Indonesia.
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Mengubah setiap kalimat dalam X menjadi sebuah vector.
        PENTING: Konversi X ke list untuk menghindari pandas KeyError.
        """
        # === INI ADALAH PERBAIKANNYA ===
        X_as_list = list(X)
        return self.model.encode(X_as_list, show_progress_bar=True)


# --- BAGIAN 3: PELATIHAN MODEL (LEVEL 3) ---

def train_and_save_model():
    """
    Fungsi utama untuk melatih model klasifikasi intent.
    Menggabungkan fitur TF-IDF (Level 1) dan Sentence Embeddings (Level 2)
    untuk melatih classifier (Level 3).
    """
    print("1. Memuat dan menyiapkan data...")
    df = load_and_prepare_data()
    if df.empty:
        print("Tidak ada data training yang ditemukan. Pastikan direktori intent benar.")
        return

    # Memisahkan data menjadi set pelatihan dan set pengujian
    X = df['training_phrase']
    y = df['intent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"   - Total {len(df)} frasa pelatihan.")
    print(f"   - Dibagi menjadi {len(X_train)} untuk training dan {len(X_test)} untuk testing.")
    
    # Membuat pipeline machine learning
    print("\n2. Membangun pipeline machine learning...")
    
    feature_generator = FeatureUnion([
        ('tfidf_vectorizer', TfidfVectorizer(
            lowercase=True, 
            analyzer='word', 
            ngram_range=(1, 2)
        )),
        ('sentence_embeddings', SentenceTransformerEmbeddings())
    ])
    
    model_pipeline = Pipeline([
        ('features', feature_generator),
        ('classifier', LinearSVC(random_state=42))
    ])

    print("3. Melatih model... (Ini mungkin butuh beberapa menit saat pertama kali)")
    model_pipeline.fit(X_train, y_train)
    
    print("\n4. Mengevaluasi model...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   - Akurasi pada data test: {accuracy:.2%}")
    # print("\nLaporan Klasifikasi Rinci:\n", classification_report(y_test, y_pred, zero_division=0))

    print("\n5. Menyimpan model dan respons ke file...")
    joblib.dump(model_pipeline, 'intent_model.pkl')
    print("   - Model berhasil disimpan sebagai 'intent_model.pkl'")
    print("   - Respons berhasil disimpan sebagai 'responses.json'")

    print("\n✅ Pelatihan selesai!")

# --- BAGIAN 4: KELAS PREDIKSI UNTUK PENGGUNAAN ---

class IntentPredictor:
    """
    Kelas untuk memuat model yang sudah dilatih dan melakukan prediksi
    pada input teks baru.
    """
    def __init__(self, model_path='intent_model.pkl', responses_path='responses.json'):
        try:
            self.model = joblib.load(model_path)
            with open(responses_path, 'r', encoding='utf-8') as f:
                self.responses = json.load(f)
        except FileNotFoundError:
            print("Error: File model atau respons tidak ditemukan. Jalankan fungsi train_and_save_model() terlebih dahulu.")
            self.model = None
            self.responses = None

    def predict(self, text, confidence_threshold=0.2):
        """
        Memprediksi intent dari sebuah kalimat.
        """
        if not self.model:
            return "Model belum dilatih. Silakan jalankan proses training.", "error"
            
        predicted_intent = self.model.predict([text])[0]
        confidence_scores = self.model.decision_function([text])
        confidence = np.max(confidence_scores)
        
        print(f"DEBUG: Teks='{text}', Prediksi='{predicted_intent}', Skor Keyakinan='{confidence:.4f}'")

        if confidence < confidence_threshold:
            # Menggunakan intent fallback khusus jika ada, jika tidak gunakan pesan default
            fallback_intent = "Default Fallback Intent"
            response_list = self.responses.get(fallback_intent, ["Maaf, saya tidak begitu mengerti. Bisa coba gunakan kalimat lain?"])
            return random.choice(response_list), fallback_intent
        
        response_list = self.responses.get(predicted_intent, [])
        if not response_list:
            return "Saya menemukan maksud Anda, tapi tidak ada respons yang disiapkan untuk itu.", predicted_intent
            
        return random.choice(response_list), predicted_intent

# --- BAGIAN 5: EKSEKUSI ---

if __name__ == "__main__":
    # Jalankan pelatihan. Setelah berhasil, beri komentar lagi pada baris ini.
    train_and_save_model()

    # Mode interaktif untuk mencoba model
    print("\n--- Memulai Mode Prediksi Interaktif ---")
    print("Ketik 'exit' untuk keluar.")
    predictor = IntentPredictor()

    if predictor.model:
        while True:
            user_input = input("Anda: ")
            if user_input.lower() == 'exit':
                break
            
            response, intent = predictor.predict(user_input)
            print(f"Bot (Intent: {intent}): {response}")