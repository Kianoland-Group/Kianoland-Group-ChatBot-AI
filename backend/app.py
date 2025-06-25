import os
import random
import numpy as np
import threading
import httpx
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Discord
import discord
from discord.ext import commands

# Import kelas IntentPredictor dari engine NLP baru kita
from nlp_engine.local_nlp_level3 import IntentPredictor

# --- Inisialisasi Aplikasi FastAPI ---
app = FastAPI()

# --- Konfigurasi Path dan .env ---
load_dotenv()
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
frontend_dir = os.path.join(project_root, "frontend")
nlp_engine_dir = os.path.join(project_root, "nlp_engine")
model_path = os.path.join(nlp_engine_dir, 'intent_model.pkl')
responses_path = os.path.join(nlp_engine_dir, 'responses.json')

# --- Memuat Model NLP ---
try:
    predictor = IntentPredictor(model_path=model_path, responses_path=responses_path)
    print("✅ Model NLP ('intent_model.pkl') berhasil dimuat.")
except Exception as e:
    print(f"❌ GAGAL memuat model NLP. Error: {e}")
    predictor = None

# --- Modifikasi Metode Predict ---
# Menyesuaikan metode 'predict' pada instance yang sudah dimuat
# agar mengembalikan skor keyakinan untuk digunakan di seluruh aplikasi.
def predict_with_confidence(self, text, confidence_threshold=0.2):
    if not self.model:
        return "Model belum dilatih.", "error", 0.0
    predicted_intent = self.model.predict([text])[0]
    confidence_scores = self.model.decision_function([text])
    confidence = float(np.max(confidence_scores))
    print(f"DEBUG: Teks='{text}', Prediksi='{predicted_intent}', Skor Keyakinan='{confidence:.4f}'")
    if confidence < confidence_threshold:
        fallback_intent = "default_fallback"
        response_list = self.responses.get(fallback_intent, ["Maaf, saya tidak begitu mengerti. Bisa coba gunakan kalimat lain?"])
        return random.choice(response_list), fallback_intent, confidence
    response_list = self.responses.get(predicted_intent, [])
    if not response_list:
        return "Saya menemukan maksud Anda, tapi tidak ada respons yang disiapkan untuk itu.", predicted_intent, confidence
    return random.choice(response_list), predicted_intent, confidence

if predictor:
    predictor.predict = predict_with_confidence.__get__(predictor, IntentPredictor)

# --- Middleware CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Konfigurasi Discord & Telegram ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEDICATED_CHANNEL_ID = int(os.getenv("DEDICATED_CHANNEL_ID", "0"))
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# --- Logika Bot Discord ---
intents = discord.Intents.default()
intents.message_content = True
discord_bot = commands.Bot(command_prefix="!", intents=intents)

def run_discord_bot():
    if not DISCORD_TOKEN or not DEDICATED_CHANNEL_ID:
        print("⚠️  Variabel lingkungan Discord (TOKEN/CHANNEL_ID) tidak diatur. Bot Discord tidak akan berjalan.")
        return

    @discord_bot.event
    async def on_ready():
        print(f'✅ Discord bot logged in as {discord_bot.user}')
        channel = discord_bot.get_channel(DEDICATED_CHANNEL_ID)
        if channel:
            await discord_bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=f"#{channel.name}"))

    @discord_bot.event
    async def on_message(message):
        if message.author.bot or message.channel.id != DEDICATED_CHANNEL_ID:
            return

        if predictor:
            # Menggunakan model NLP baru untuk merespons
            response, intent, confidence = predictor.predict(message.content)
            await message.reply(response)
        else:
            await message.reply("Maaf, model NLP sedang tidak aktif.")

    discord_bot.run(DISCORD_TOKEN)


# --- Endpoint Frontend Web ---
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model NLP tidak berhasil dimuat.")
    try:
        answer, intent, confidence = predictor.predict(request.user_input)
        debug_info = f"Intent: {intent}\nConfidence: {confidence:.4f}"
        return {"answer": answer, "debug_info": debug_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {e}")

# --- Endpoint Telegram Webhook ---
async def send_telegram_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(f"{TELEGRAM_API_URL}/sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    if not predictor:
        return {"ok": False, "error": "Model NLP tidak aktif"}
    try:
        update = await request.json()
        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            text = update["message"].get("text", "")
            if text:
                response, _, _ = predictor.predict(text)
                await send_telegram_message(chat_id, response)
        return {"ok": True}
    except Exception as e:
        print(f"Error di webhook Telegram: {e}")
        return {"ok": False}


# --- Event Startup & Health Check ---
@app.on_event("startup")
async def startup_event():
    # Jalankan bot Discord di thread terpisah
    discord_thread = threading.Thread(target=run_discord_bot, daemon=True)
    discord_thread.start()

    # Atur webhook untuk Telegram
    if TELEGRAM_TOKEN and TELEGRAM_WEBHOOK_URL:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{TELEGRAM_API_URL}/setWebhook", json={"url": TELEGRAM_WEBHOOK_URL})
                if response.status_code == 200:
                    print("✅ Webhook Telegram berhasil diatur.")
                else:
                    print(f"❌ Gagal mengatur webhook Telegram: {response.text}")
            except Exception as e:
                print(f"❌ Error saat menghubungkan ke API Telegram: {e}")
    else:
        print("⚠️  Variabel lingkungan Telegram (TOKEN/WEBHOOK_URL) tidak diatur. Integrasi Telegram tidak akan berjalan.")


@app.get("/health")
async def health_check():
    return {"status": "online", "model_loaded": predictor is not None}