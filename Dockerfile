# Gunakan image Python resmi sebagai dasar
# Menambahkan "bookworm" untuk versi Debian yang stabil
FROM python:3.12-slim-bookworm

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Instal dependensi sistem yang diperlukan: git dan git-lfs
# Ini adalah langkah KRUSIAL untuk mengunduh model dari Git LFS
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install

# Salin file requirements.txt terlebih dahulu
# Ini memanfaatkan caching Docker: layer ini hanya akan di-build ulang jika requirements.txt berubah
COPY ./requirements.txt /app/requirements.txt

# Instal dependensi Python dari file requirements.txt yang baru
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Salin seluruh kode aplikasi setelah dependensi terinstal
# Ini termasuk backend, frontend, dan yang terpenting, nlp_engine
COPY ./backend /app/backend
COPY ./dialogflow_kianoland /app/dialogflow_kianoland
COPY ./frontend /app/frontend
COPY ./nlp_engine /app/nlp_engine
# Salin juga file-file konfigurasi penting di root
# Baris untuk .env dihapus karena variabel harus dikelola oleh platform deployment
COPY ./.gitattributes /app/.gitattributes


# Beri tahu Docker bahwa container akan berjalan di port 8000
EXPOSE 8000

# Perintah untuk menjalankan aplikasi menggunakan Uvicorn
# 0.0.0.0 agar dapat diakses dari luar container
# Railway akan secara otomatis menyediakan variabel $PORT, tetapi 8000 adalah default yang baik
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]