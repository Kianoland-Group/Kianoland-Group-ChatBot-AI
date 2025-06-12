# Gunakan image Python resmi sebagai dasar
FROM python:3.12-slim-bookworm

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Salin folder backend dan data Dialogflow
COPY ./backend /app/backend
COPY ./dialogflow_kianoland /app/dialogflow_kianoland

# Instal dependensi Python
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Salin folder frontend
COPY ./frontend /app/frontend

# Beri tahu Docker bahwa container akan berjalan di port 8000
# Railway akan menggantinya dengan port yang benar secara otomatis
EXPOSE 8000

# Perintah untuk menjalankan aplikasi menggunakan Uvicorn
# Railway akan menyediakan variabel $PORT
# Menggunakan 0.0.0.0 agar dapat diakses dari luar container
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]