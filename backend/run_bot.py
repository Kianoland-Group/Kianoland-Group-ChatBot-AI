# backend/run_bot.py
# File ini KHUSUS untuk menjalankan bot Discord di layanan Background Worker Render.

import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
# Di Render, ini akan dimuat dari Environment Variables yang Anda atur di dasbor.
load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Pastikan token ada sebelum melanjutkan
if not DISCORD_BOT_TOKEN:
    print("Error: DISCORD_BOT_TOKEN tidak ditemukan di environment variables.")
    print("Bot tidak akan dijalankan.")
    exit()

# Menentukan 'intents' yang diperlukan oleh bot.
# Ini adalah izin yang diperlukan bot untuk membaca pesan, melihat anggota server, dll.
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True

# Membuat instance bot dengan prefix perintah "!" dan intents yang sudah ditentukan.
bot = commands.Bot(command_prefix="!", intents=intents)

# Event handler yang akan dipanggil ketika bot berhasil terhubung ke Discord.
@bot.event
async def on_ready():
    """Event yang dijalankan saat bot online dan siap."""
    print(f'Bot {bot.user} telah berhasil terhubung ke Discord!')
    print('Status: Online')
    print('-----------------------------------------')

# Contoh sebuah perintah sederhana untuk bot.
@bot.command(name='hello')
async def hello(ctx):
    """Perintah sederhana untuk mengetes apakah bot merespons."""
    await ctx.send(f'Halo, {ctx.author.name}! Saya adalah Kianoland Bot.')

# Contoh perintah lain
@bot.command(name='info')
async def info(ctx):
    """Memberikan informasi tentang server."""
    server_name = ctx.guild.name
    member_count = ctx.guild.member_count
    await ctx.send(f'Server ini bernama "{server_name}" dan memiliki {member_count} anggota.')

# Anda bisa menambahkan lebih banyak event dan command di sini
# @bot.event
# async def on_message(message):
#     # Jangan lupa untuk memproses perintah juga
#     await bot.process_commands(message)


# Entry point untuk menjalankan bot.
if __name__ == "__main__":
    print("Mencoba memulai Discord bot...")
    # Menjalankan bot dengan token yang telah diambil.
    # Kode ini akan berjalan terus-menerus sampai prosesnya dihentikan.
    bot.run(DISCORD_BOT_TOKEN)
