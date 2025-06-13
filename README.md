# Kianoland Group ChatBot AI 🤖

A multi-platform chatbot for property consultations, integrated with Any Website Discord and Telegram using Hybrid AI with NLP and Rule-Based System.

## Fitur Utama

- Menjawab pertanyaan tentang proyek properti
- Memberikan informasi harga, fasilitas, dan lokasi
- Rekomendasi proyek berdasarkan lokasi
- Sistem konsultasi privat di Discord
- Tersedia di Discord, Telegram, dan Web

## Struktur Proyek

```bash
├── backend/
│ ├── app.py                # Aplikasi utama (FastAPI)
│ ├── local_nlp.py          # Modul NLP lokal
│ └── requirements.txt      # Dependensi Python
├── dialogflow_kianoland/   # Data pelatihan Dialogflow
│ ├── entities/             # Entitas sistem
│ └── intents/              # Intents dialog
├── frontend/               # Antarmuka web chatbot
│ ├── index.html
│ ├── script.js
│ └── style.css
└── .env                    # Konfigurasi lingkungan
```

## Instalasi

1. Clone repositori:

```bash
git clone https://github.com/Kianoland-Group/Kianoland-Group-ChatBot-AI.git
cd Kianoland-Group-ChatBot-AI
```

2. Buat virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instal dependensi:

```bash
pip install -r backend/requirements.txt
```

4. Buat file `.env` dengan konten:

```env
DISCORD_TOKEN=your-bot-discord-token
TELEGRAM_TOKEN=your-bot-telegram-token
TELEGRAM_WEBHOOK_URL=your-ngrok-url/telegram-webhook
DEDICATED_CHANNEL_ID=your-discord-channel-id
```

## Menjalankan Aplikasi

1. Jalankan backend:

```bash
uvicorn backend.app:app --reload --port 8000
```

2. Buka frontend:

- Buka file `frontend/index.html` di browser

3. Untuk Discord bot:

- Bot akan otomatis berjalan setelah backend dijalankan

4. Untuk Telegram:

- Pastikan webhook sudah terdaftar di `TELEGRAM_WEBHOOK_URL`

## Endpoint API

- `POST /detect-intent` - Deteksi intent dari teks
- `POST /chat` - Endpoint chat untuk web
- `POST /telegram-webhook` - Webhook Telegram
- `POST /discord-webhook` - Webhook Discord
- `GET /health` - Health check

## Kontribusi

1. Fork repositori
2. Buat branch fitur baru (`git checkout -b fitur-baru`)
3. Commit perubahan (`git commit -am 'Tambahkan fitur'`)
4. Push ke branch (`git push origin fitur-baru`)
5. Buat pull request

## Lisensi

[MIT]()
