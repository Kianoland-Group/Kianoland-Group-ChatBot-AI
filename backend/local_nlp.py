import json
import os
from difflib import SequenceMatcher
from typing import Dict, List, Optional
import re

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIALOGFLOW_FOLDER = os.path.join(BASE_DIR, "dialogflow_kianoland")
ENTITIES_FOLDER = os.path.join(DIALOGFLOW_FOLDER, "entities")
INTENTS_FOLDER = os.path.join(DIALOGFLOW_FOLDER, "intents")

# Data storage
INTENTS: List[dict] = []
ENTITIES: Dict[str, list] = {}

def load_resources():
    """Load all NLP resources"""
    load_intents()
    load_entities()

marketing_contacts = {
    "Natureland Kiano 2": "Bp. Toni - 0896 3823 0725",
    "Natureland Kiano 3": "Bp. Toni - 0896 3823 0725",
    "Green Jonggol Village": "0851-7955-3681"
}

def load_intents():
    """Load intents from JSON files"""
    global INTENTS
    INTENTS = []
    
    for filename in os.listdir(INTENTS_FOLDER):
        if filename.endswith('.json'):
            with open(os.path.join(INTENTS_FOLDER, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'displayName' in data:
                    # Ekstrak frasa pelatihan dengan normalisasi
                    training_phrases = []
                    for phrase in data.get('trainingPhrases', []):
                        full_phrase = ""
                        for part in phrase['parts']:
                            # NORMALISASI: hilangkan spasi berlebih dan perbaiki huruf berulang
                            normalized_text = re.sub(r'\s+', ' ', part['text']).strip()
                            normalized_text = re.sub(r'(\w)\1{2,}', r'\1', normalized_text.lower())
                            full_phrase += normalized_text
                        training_phrases.append(full_phrase)
                    
                    # Ekstrak respons
                    responses = []
                    for message in data.get('messages', []):
                        if 'text' in message:
                            combined_text = '\n'.join(
                                line.strip().rstrip(',') for line in message['text']['text'] if line.strip()
                            )
                            responses.append(combined_text)

                    
                    INTENTS.append({
                        'name': data['displayName'],
                        'phrases': training_phrases,
                        'responses': responses
                    })
    print(f"✅ Loaded {len(INTENTS)} intents")
    # Debug: Tampilkan nama intent dan jumlah frasa
    for intent in INTENTS:
        print(f"  - {intent['name']} ({len(intent['phrases'])} phrases)")

def load_entities():
    """Load entities from JSON files"""
    global ENTITIES
    ENTITIES = {}
    
    for filename in os.listdir(ENTITIES_FOLDER):
        if filename.endswith('_entries.json'): 
            entity_name = filename.replace('_entries.json', '')
            with open(os.path.join(ENTITIES_FOLDER, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                ENTITIES[entity_name] = data['entries'] if isinstance(data, dict) else data
    print(f"✅ Loaded {len(ENTITIES)} entities")

# Tambahkan fungsi similar setelah load_entities
def similar(a: str, b: str) -> float:
    """Menghitung similarity ratio antara dua string (case-insensitive)"""
    a_low = a.lower()
    b_low = b.lower()
    return SequenceMatcher(None, a_low, b_low).ratio()

# 1. Dapatkan daftar proyek valid dari entitas
def get_valid_projects():
    """Mendapatkan daftar proyek valid yang tersedia (tanpa Kiano 1 dan 2)"""
    available_projects = [
        "Natureland Kiano 3",
        "Green Jonggol Village"
    ]
    return available_projects

# 2. Fungsi validasi proyek menggunakan entitas
def is_valid_project(project_name: str) -> bool:
    """Cek apakah proyek valid berdasarkan entitas secara lebih ketat."""
    project_lower = project_name.lower().strip()
    
    # Cek di semua entitas proyek
    for entry in ENTITIES.get('proyek', []):
        # Cek value utama (nama resmi proyek)
        if project_lower == entry['value'].lower():
            return True
        
        # Cek semua sinonimnya
        for synonym in entry.get('synonyms', []):
            if project_lower == synonym.lower():
                return True
                
    return False

def detect_entities(text: str) -> Dict[str, str]:
    """Detects projects and locations from user input using entity lists."""
    detected = {}
    text_lower = text.lower()

    project_stop_words = [
        'proyek', 'project', 'properti', 'rumah', 'perumahan', 'yang', 'ada', 
        'tersedia', 'apa', 'saja', 'aja', 'semua', 'list', 'daftar', 'informasi',
        'info', 'harga', 'promo', 'fasilitas', 'lokasi'
    ]

    # --- Deteksi Proyek (Logika ini tetap sama) ---
    for entry in ENTITIES.get('proyek', []):
        for synonym in entry.get('synonyms', []):
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                if synonym.lower() not in project_stop_words:
                    detected['proyek'] = entry['value']
                    print(f"✅ Proyek terdeteksi (dari entitas): '{synonym}' → {entry['value']}")
                    break
        if 'proyek' in detected:
            break
            
    if 'proyek' not in detected:
        match = re.search(r'\b(kiano|nlk)\s*(\d+)\b', text_lower)
        if match:
            project_base_name = "Natureland Kiano"
            project_number = match.group(2)
            constructed_name = f"{project_base_name} {project_number}"
            detected['proyek'] = constructed_name
            print(f"⚠️  Proyek terdeteksi via pola (mungkin tidak valid): '{match.group(0)}' → {constructed_name}")

    # --- PERBAIKAN TOTAL: Deteksi Lokasi dari Entitas ---
    # Logika lama yang salah telah dihapus dan diganti dengan yang ini.
    for entry in ENTITIES.get('lokasi', []):
        for synonym in entry.get('synonyms', []):
            # Menggunakan word boundaries (\b) untuk pencocokan yang akurat
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                detected['lokasi'] = entry['value']
                print(f"✅ Lokasi terdeteksi (dari entitas): '{synonym}' → {entry['value']}")
                break
        if 'lokasi' in detected:
            break

    print(f"🧩 Entitas terdeteksi final: {detected}")
    return detected

def detect_intent_local(user_input: str) -> Dict[str, str]:
    """Detect intent using a final, robust, rule-based priority system."""
    user_input_normalized = re.sub(r'(\w)\1{2,}', r'\1', user_input.lower().strip())
    print(f"\n🔍 User input: '{user_input}' → Normalized: '{user_input_normalized}'")

    entities = detect_entities(user_input)
    project = entities.get('proyek')
    lokasi = entities.get('lokasi')

    # ===== ATURAN #1A: TANGANI PROYEK YANG TIDAK ADA SAMA SEKALI (contoh: Kiano 4) =====
    if project and not is_valid_project(project):
        print(f"🎯 ATURAN #1A: Proyek Tidak Dikenal '{project}' Terdeteksi.")
        return format_response(
            f"Maaf, proyek '{project}' tidak ada atau tidak tersedia di Kianoland Group.\n\n"
            f"Proyek yang tersedia saat ini:\n• Natureland Kiano 3\n• Green Jonggol Village"
        )

    # ===== ATURAN #1B: TANGANI PROYEK YANG ADA TAPI SUDAH SOLD OUT (contoh: Kiano 1) =====
    sold_out_projects = ["Natureland Kiano 1", "Natureland Kiano 2"]
    is_asking_lokasi = 'lokasi' in user_input_normalized or 'alamat' in user_input_normalized
    if project in sold_out_projects and not is_asking_lokasi:
        print(f"🎯 ATURAN #1B: Proyek Sold Out '{project}' Terdeteksi.")
        return format_response(
            f"Maaf, proyek {project} sudah sold out. Kami merekomendasikan proyek terbaru kami:\n\n"
            f"🏡 Natureland Kiano 3 (Cibarusah, Bekasi)\n🌳 Green Jonggol Village (Jonggol, Bogor)\n\n"
            f"Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
        )

    # ===== ATURAN #2: PERTANYAAN SPESIFIK BERDASARKAN KATA KUNCI =====
    specific_keywords_map = {
        'info_promo': ['promo', 'diskon', 'dp', 'uang muka'],
        'info_harga': ['harga', 'cicilan', 'angsuran', 'biaya'],
        'info_fasilitas': ['fasilitas'],
        'info_lokasi': ['lokasi', 'alamat', 'peta', 'letak'],
        'syarat_dokumen': ['syarat', 'persyaratan', 'dokumen', 'kpr'],
        'bantuan': ['bantuan', 'panduan', 'tolong', 'bantu', 'tidak mengerti', 'tidak paham', 'bingung'],
        'minat_beli': ['beli', 'minat', 'lihat']
    }
    
    for intent_name, keywords in specific_keywords_map.items():
        if any(kw in user_input_normalized for kw in keywords):
            if intent_name == 'info_promo' and not project:
                print("🎯 ATURAN #2.A: Permintaan Promo Umum Terdeteksi.")
                promo_intent = next((i for i in INTENTS if i['name'] == 'info_promo'), None)
                if promo_intent:
                    response_text = process_conditional_templates(promo_intent['responses'][0], project='all_promos')
                    return format_response(response_text)
            
            print(f"🎯 ATURAN #2.B: Intent Spesifik '{intent_name}' Terdeteksi.")
            forced_intent = next((i for i in INTENTS if i['name'] == intent_name), None)
            if forced_intent:
                response_text = process_conditional_templates(forced_intent['responses'][0], project, lokasi)
                return format_response(response_text)

    # ===== ATURAN #2C (BARU): INFO PROYEK VALID (contoh: info kiano 3) =====
    # Jika ada proyek valid terdeteksi DAN tidak ada kata kunci spesifik lain,
    # maka asumsikan pengguna ingin info umum proyek tersebut.
    if project and project not in sold_out_projects:
        print(f"🎯 ATURAN #2C: Permintaan Info Umum untuk Proyek Valid '{project}'.")
        info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
        if info_intent:
            response_text = process_conditional_templates(info_intent['responses'][0], project)
            return format_response(response_text)

    # ===== ATURAN #3: RUMAH SUBSIDI & KOMERSIL =====
    if 'subsidi' in user_input_normalized or 'komersil' in user_input_normalized:
        project = "Green Jonggol Village"
        info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
        if info_intent:
            intro_text = "Untuk rumah subsidi, kami merekomendasikan **Green Jonggol Village**.\n\nBerikut informasinya:\n" if 'subsidi' in user_input_normalized else "Untuk rumah komersil, kami merekomendasikan **Green Jonggol Village**.\n\nBerikut informasinya:\n"
            processed_response = process_conditional_templates(info_intent['responses'][0], project=project)
            return format_response(intro_text + processed_response)

    # ===== ATURAN #4: REKOMENDASI LOKASI =====
    rekomendasi_keywords = ['rekomendasi', 'rekom', 'sarankan', 'saran', 'cocok', 'rumah', 'proyek', 'properti', 'hunian']
    if lokasi and any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print(f"🎯 ATURAN #4A: Rekomendasi untuk Lokasi Dikenal '{lokasi}' Terdeteksi.")
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None)
        if rekomendasi_intent:
            response_text = process_conditional_templates(rekomendasi_intent['responses'][0], lokasi=lokasi)
            return format_response(response_text)
    elif not lokasi and any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print("🎯 ATURAN #4B: Rekomendasi untuk Lokasi Tidak Dikenal Terdeteksi.")
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None)
        if rekomendasi_intent:
            response_text = process_conditional_templates(rekomendasi_intent['responses'][0], lokasi="tersebut")
            return format_response(response_text)

    # ===== ATURAN #5: PENCOCOKAN KEMIRIPAN UMUM (FALLBACK) =====
    print("🚦 Lanjut ke Aturan #5: Pencocokan Berbasis Kemiripan.")
    # (Kode ini tetap ada sebagai jaring pengaman terakhir sebelum fallback total)
    best_match = None
    highest_score = 0.75
    for intent in INTENTS:
        if intent['name'] in ['default_fallback', 'info_promo', 'info_harga', 'info_fasilitas', 'info_lokasi', 'syarat_dokumen', 'bantuan', 'rekomendasi_proyek']:
            continue
        for phrase in intent.get('phrases', []):
            similarity = SequenceMatcher(None, user_input_normalized, phrase).ratio()
            if similarity > highest_score:
                highest_score = similarity
                best_match = intent
    if best_match:
        print(f"🎯 Best match by similarity: {best_match['name']} (score: {highest_score:.2f})")
        response_text = process_conditional_templates(best_match['responses'][0], project)
        return format_response(response_text)
    
    # ===== ATURAN #6: FALLBACK TERAKHIR =====
    print("🛑 Fallback Terakhir.")
    fallback_intent = next((i for i in INTENTS if i['name'] == 'default_fallback'), None)
    if fallback_intent:
        return format_response(fallback_intent['responses'][0])
    return format_response("Maaf, saya tidak dapat memproses permintaan Anda saat ini.")

def process_conditional_templates(text: str, project: str = None, lokasi: str = None, primary: str = None, secondary: str = None) -> str:
    """Process conditional templates with intelligent block selection based on project or location."""

    # --- Tentukan kunci selektor utama (bisa proyek atau lokasi) ---
    selector = primary or project or lokasi

    # 1. Coba cari blok spesifik menggunakan selektor
    if selector:
        escaped_selector = re.escape(selector)
        pattern = r'\{\{#' + escaped_selector + r'\}\}(.*?)\{\{/' + escaped_selector + r'\}\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Ganti placeholder di dalam konten yang sudah dipilih
            content = content.replace("{{proyek}}", project if project else "")
            content = content.replace("{{lokasi}}", lokasi if lokasi else "")
            return content

    # 2. Jika tidak ada blok spesifik, coba cari blok fallback
    fallback_pattern = r'\{\{#fallback\}\}(.*?)\{\{/fallback\}\}'
    fallback_match = re.search(fallback_pattern, text, re.DOTALL)
    if fallback_match:
        fallback_text = fallback_match.group(1).strip()
        # Ganti placeholder di dalam teks fallback
        fallback_text = fallback_text.replace("{{proyek}}", project if project else "")
        fallback_text = fallback_text.replace("{{lokasi}}", lokasi if lokasi else "")
        return fallback_text

    # 3. Jika tidak ada blok yang cocok sama sekali, bersihkan template dan kembalikan teks asli
    text = re.sub(r'\{\{#[^}]+\}\}', '', text)
    text = re.sub(r'\{\{/[^}]+\}\}', '', text)
    return text.strip()

def format_response(text: str) -> Dict[str, str]:
    """Format response for all platforms"""
    # Clean special characters
    text = text.replace('\\n', '\n').replace('\\"', '"')
    
    return {
        'raw': text,
        'discord': text.replace('bold_start', '**').replace('bold_end', '**'),
        'telegram': text.replace('**', '').replace('bold_start', '<b>').replace('bold_end', '</b>'),
        'web': text.replace('**', '').replace('bold_start', '<strong>').replace('bold_end', '</strong>')
    }

# Initialize on import
load_resources()