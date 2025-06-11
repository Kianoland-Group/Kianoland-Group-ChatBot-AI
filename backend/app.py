from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
from local_nlp import detect_intent_local as detect_intent, load_intents
import discord
from discord.ext import commands
import asyncio
import threading
from dotenv import load_dotenv
import os

# 🚀 Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kianolandgroup.com", 
        "https://www.kianolandgroup.com",
        "https://kianolandgroup.netlify.app", 
        "http://localhost:8000",  # Untuk development
        "https://79c8-157-15-46-172.ngrok-free.app",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True
)

# 🛠️ Configurations from .env
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEDICATED_CHANNEL_ID = int(os.getenv("DEDICATED_CHANNEL_ID"))
BOT_PREFIXES = ('!', '/', '$')

@app.post("/detect-intent")
async def detect_intent_endpoint(text: str):
    return detect_intent(text)

@app.get("/")
async def root():
    return {"message": "Kiano Property Bot API is running"}

# Initialize Discord Client
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
discord_bot = commands.Bot(command_prefix=BOT_PREFIXES, intents=intents)

# Discord Models
class DiscordMessage(BaseModel):
    content: str
    channel_id: int
    author: dict

# Discord Bot Functions
def run_discord_bot():
    @discord_bot.event
    async def on_ready():
        await discord_bot.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name=f"#{discord_bot.get_channel(DEDICATED_CHANNEL_ID).name}"
            )
        )
        print(f'Logged in as {discord_bot.user} (ID: {discord_bot.user.id})')

        channel = discord_bot.get_channel(DEDICATED_CHANNEL_ID)
        async for msg in channel.history(limit=5):
            if msg.author == discord_bot.user and "PANDUAN" in msg.content:
                break
        else:
            await channel.send(
                "📌 **PANDUAN PENGGUNAAN**\n"
                "1. Ketik `!info` untuk melihat promo properti\n"
                "2. Gunakan `!konsul [pertanyaan]` untuk bantuan\n"
                "3. Bot hanya aktif di channel ini"
            )

    @discord_bot.event
    async def on_message(message):
        if message.author.bot:
            return

        # Handle commands first
        ctx = await discord_bot.get_context(message)
        if ctx.command:
            await discord_bot.invoke(ctx)
            return

        try:
            if message.channel.id == DEDICATED_CHANNEL_ID:
                response = detect_intent(message.content)
                if not response or 'discord' not in response:
                    await message.reply("Maaf, terjadi kesalahan saat memproses permintaan Anda")
                else:
                    # Pecah pesan dan kirim satu per satu
                    messages_to_send = response['discord'].split('|||')
                    for msg in messages_to_send:
                        if msg.strip(): # Pastikan pesan tidak kosong
                            await message.reply(msg.strip())
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            await message.reply("Maaf, terjadi kesalahan. Silakan coba lagi.")

    @discord_bot.command()
    async def proyek(ctx):
        result = detect_intent("daftar proyek")
        await ctx.send(result['discord'])

    @discord_bot.command()
    async def info(ctx):
        response = detect_intent("info properti")
        await ctx.send(response['discord'])

    @discord_bot.command()
    async def konsul(ctx, *, question: str = None):
        if not question:
            await ctx.send("Silakan ajukan pertanyaan Anda setelah perintah `!konsul`. Contoh: `!konsul info harga Kiano 3`")
            return

        thread = await ctx.channel.create_thread(
            name=f"Konsul-{ctx.author.display_name}",
            type=discord.ChannelType.private_thread,
            reason=f"Konsultasi properti oleh {ctx.author}"
        )
        response = detect_intent(question)
        await thread.send(
            f"🛎️ Konsultasi dimulai oleh {ctx.author.mention}!\n"
            f"**Pertanyaan:** {question}\n\n"
            f"**Jawaban:** {response['discord']}"
        )
        await ctx.message.delete()

    discord_bot.run(DISCORD_TOKEN)

@app.on_event("startup")
async def startup_event():
    load_intents()
    thread = threading.Thread(target=run_discord_bot, daemon=True)
    thread.start()
    
    # Pindahkan inisialisasi Telegram webhook ke sini
    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{TELEGRAM_API_URL}/setWebhook",
            json={"url": webhook_url}
        )
        print("Telegram setWebhook result:", res.json())

# REST API Endpoints
@app.post("/discord-webhook")
async def discord_webhook(message: DiscordMessage):
    try:
        if message.author.get("bot", False):
            return {"status": "ignored"}

        result = detect_intent(message.content)
        channel = discord_bot.get_channel(message.channel_id)
        await channel.send(result['discord'])

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(400, str(e))

# 🛠️ Konfigurasi Telegram
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

async def send_telegram_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        )

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = detect_intent(request.user_input)
        # Pecah respons menjadi beberapa pesan jika ada pemisah '|||'
        formatted_responses = result['web'].split('|||')
        
        return {
            "response": {
                "raw": result['raw'],
                "formatted": formatted_responses # Kirim sebagai list
            }
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    try:
        update = await request.json()
        print("Received Telegram update:", update)  # Log update
        
        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            text = update["message"].get("text", "")

            result = detect_intent(text)

            # Pecah pesan dan kirim satu per satu
            messages_to_send = result['telegram'].split('|||')
            for msg in messages_to_send:
                if msg.strip(): # Pastikan pesan tidak kosong
                    await send_telegram_message(chat_id, msg.strip())
            
        return {"ok": True}
    except Exception as e:
        print("Error in telegram_webhook:", str(e))  # Log error
        raise HTTPException(500, "Internal Server Error")

@app.on_event("startup")
async def on_startup():
    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{TELEGRAM_API_URL}/setWebhook",
            json={"url": webhook_url}
        )
        print("setWebhook result:", res.json())

@app.get("/health")
async def health_check():
    return {"status": "online"}