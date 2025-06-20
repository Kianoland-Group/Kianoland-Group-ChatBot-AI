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
    "default": "+62 811-1611-724"
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
    """Detects projects, locations, and house types from user input."""
    detected = {}
    text_lower = text.lower()

    # --- Deteksi Proyek ---
    # Prioritaskan deteksi proyek dengan sinonim yang lebih spesifik
    for entry in ENTITIES.get('proyek', []):
        for synonym in entry.get('synonyms', []):
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                # Pastikan ini bukan sinonim umum seperti 'proyek' atau 'rumah'
                if synonym.lower() not in ['proyek', 'project', 'properti', 'rumah', 'perumahan']:
                    detected['proyek'] = entry['value']
                    print(f"✅ Proyek terdeteksi (from entity): '{synonym}' -> {entry['value']}")
                    break
        if 'proyek' in detected:
            break
            
    # Fallback untuk deteksi proyek jika tidak ditemukan via entitas langsung (misal: "kiano 3")
    if 'proyek' not in detected:
        match = re.search(r'\b(kiano|nlk)\s*(\d+)\b', text_lower)
        if match:
            project_base_name = "Natureland Kiano"
            project_number = match.group(2)
            constructed_name = f"{project_base_name} {project_number}"
            # Hanya tambahkan jika itu bukan pertanyaan umum seperti "kiano berapa"
            if constructed_name.lower() not in ['natureland kiano group']: 
                detected['proyek'] = constructed_name
                print(f"⚠️  Proyek terdeteksi via pattern (might be invalid): '{match.group(0)}' -> {constructed_name}")

    # --- Deteksi Lokasi ---
    for entry in ENTITIES.get('lokasi', []):
        for synonym in entry.get('synonyms', []):
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                detected['lokasi'] = entry['value']
                print(f"✅ Lokasi terdeteksi (from entity): '{synonym}' -> {entry['value']}")
                break
        if 'lokasi' in detected:
            break

    # --- Deteksi Tipe Rumah (umum, misal "30/60") ---
    for entry in ENTITIES.get('tipe_rumah', []):
        for synonym in entry.get('synonyms', []):
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                detected['tipe_rumah'] = entry['value']
                print(f"✅ Tipe Rumah terdeteksi: '{synonym}' -> {key}")
                break
        if 'tipe_rumah' in detected:
            break

    # --- Deteksi Tipe Rumah Kiano 3 (Manual - untuk selector di template) ---
    kiano3_types = {
        'K3_1_Lantai': ['1 lantai', 'satu lantai', '40/60', '40 per 60', 'empat puluh / enam puluh', 'empat puluh per enam puluh'],
        'K3_Mezzanine': ['mezzanine', '1,5 lantai', 'satu setengah lantai', '60/60', '1 koma 5 lantai', 'satu koma lima lantai', '60 per 60', 'enam puluh / enam puluh', 'enam puluh per enam puluh'],
        'K3_2_Lantai': ['2 lantai', 'dua lantai', '80/60', '80 per 60', 'delapan puluh / enam puluh', 'delapan puluh per enam puluh']
    }
    for key, synonyms in kiano3_types.items():
        for synonym in synonyms:
            if synonym in text_lower:
                detected['tipe_kiano3'] = key
                print(f"✅ Tipe Kiano 3 terdeteksi (manual): '{synonym}' -> {key}")
                break 
        if 'tipe_kiano3' in detected:
            break
            
    # --- Deteksi Tipe Green Jonggol Village (Manual - untuk selector di template) ---
    gjv_types = {
        'GJV_subsidi': ['subsidi', 'tipe 30/60', '30/60', '30 per 60', 'tiga puluh / enam puluh', 'tiga puluh per enam puluh'],
        'GJV_komersil': ['komersil', 'tipe 36/72', '36/72', '36 per 72', 'tiga enam / tujuh dua', 'tiga enam per tujuh dua']
    }
    for key, synonyms in gjv_types.items():
        for synonym in synonyms:
            if synonym in text_lower:
                detected['tipe_gjv'] = key
                print(f"✅ Tipe GJV terdeteksi (manual): '{synonym}' -> {key}")
                break
        if 'tipe_gjv' in detected:
            break


    print(f"🧩 Detected entities: {detected}")
    return detected

def detect_intent_local(user_input: str) -> Dict[str, str]:
    """Detect intent using a final, robust, rule-based priority system."""
    user_input_normalized = re.sub(r'(\w)\1{2,}', r'\1', user_input.lower().strip())
    print(f"\n🔍 User input: '{user_input}' -> Normalized: '{user_input_normalized}'")

    entities = detect_entities(user_input)
    project = entities.get('proyek')
    lokasi = entities.get('lokasi')
    tipe_rumah = entities.get('tipe_rumah')
    tipe_kiano3 = entities.get('tipe_kiano3')
    tipe_gjv = entities.get('tipe_gjv')

    # Handle Discord-specific !info command explicitly at the beginning if needed
    if user_input_normalized == '!info':
        print(f"🎯 ATURAN #0 (Discord Command): '!info' detected. Triggering 'daftar_proyek' intent.")
        daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
        if daftar_intent:
            return format_response(daftar_intent['responses'][0])
            
    # ===== NEW ATURAN #1 (INFO KONTAK) - PRIORITAS SANGAT TINGGI UNTUK PERMINTAAN KONTAK EKPLISIT =====
    kontak_keywords = [
        'kontak', 'cs', 'customer service', 'admin', 'telepon', 'nomor', 'hubungi', 'hp', 'telp',
        'contact us', 'bicara dengan orang', 'wa marketing', 'kantor kianoland', 'email kianoland', 'email', 'wa',
        'berapa nomor', 'nomor berapa', 'nomor telepon', 'nomor hp', 'nomor wa', 'berikan nomor telepon', 
        'berikan nomor hp', 'telepon kianoland', 'telepon marketing', 'telepon admin',
        'email marketing', 'email admin'
    ] 
    # Pengecualian: jika ada "alamat" yang sangat spesifik, biarkan jatuh ke aturan lokasi.
    if any(kw in user_input_normalized for kw in kontak_keywords):
        if 'alamat' in user_input_normalized and ('kantor' in user_input_normalized or 'lokasi' in user_input_normalized):
            pass # Biarkan jatuh ke aturan lokasi
        else:
            kontak_intent = next((i for i in INTENTS if i['name'] == 'info_kontak'), None)
            if kontak_intent:
                print("🎯 NEW ATURAN #1 (Info Kontak): Explicit contact request detected. Triggering 'info_kontak' intent.")
                return format_response(kontak_intent['responses'][0])

    # ===== NEW ATURAN #2 (MINAT BELI) - Paling Prioritas setelah info kontak =====
    minat_beli_keywords = [
        'saya ingin beli rumah', 'saya mau booking unit', 'ingin beli rumahnya',
        'bagaimana cara pembayarannya', 'langkah selanjutnya untuk pembelian apa',
        'saya minat serius', 'bisa jadwalkan kunjungan ke lokasi',
        'bagaimana cara bayar booking fee', 'saya siap untuk proses kpr',
        'hubungkan saya dengan marketing', 'saya mau mulai proses pembelian',
        'mau beli rumah', 'cara beli rumah di sini gimana', 'saya minat beli',
        'saya mau ambil satu unit', 'bagaimana cara booking', 'jadwalkan saya untuk survey',
        'ok saya deal', 'selanjutnya gimana', 'minta kontak salesnya dong',
        'saya mau lanjut', 'proses lebih lanjut gimana', 'bisa bantu proses pembelian',
        'gimana proses pembayaran', 'tahapan pembayaran rumah', 'sistem pembayaran di kianoland',
        'cara pembayaran rumah', 'metode pembayaran', 'tahapan bayar rumah',
        'langkah-langkah pembelian', 'prosedur pembelian rumahnya gimana', 'jelaskan alur pembelian'
    ]
    if any(kw in user_input_normalized for kw in minat_beli_keywords):
        print(f"🎯 NEW ATURAN #2 (Minat Beli): Explicit buying/process intent keyword detected. Triggering 'minat_beli' intent.")
        minat_beli_intent = next((i for i in INTENTS if i['name'] == 'minat_beli'), None)
        if minat_beli_intent:
            return format_response(minat_beli_intent['responses'][0])

    # ===== NEW ATURAN #3 (SYARAT DOKUMEN) - Prioritas Tinggi setelah minat beli =====
    syarat_dokumen_keywords = [
        'syarat', 'persyaratan', 'dokumen', 'kpr', 'berkas', 'prosedur kredit',
        'proses kredit pemilikan rumah', 'saya perlu siapkan apa saja untuk beli rumah'
    ]
    if any(kw in user_input_normalized for kw in syarat_dokumen_keywords):
        print(f"🎯 NEW ATURAN #3 (Syarat Dokumen): Explicit document requirement keyword detected. Triggering 'syarat_dokumen' intent.")
        syarat_dokumen_intent = next((i for i in INTENTS if i['name'] == 'syarat_dokumen'), None)
        if syarat_dokumen_intent:
            return format_response(syarat_dokumen_intent['responses'][0])

    # ===== NEW ATURAN #4 (Bantuan/Help) - Setelah yang lebih spesifik =====
    help_keywords = ['bantuan', 'panduan', 'cara pakai', 'menu', 'apa saja yang bisa ditanyakan', 'saya perlu bantuan', 'tolong bantu saya', 'bagaimana cara menggunakan bot ini', 'saya butuh panduan', 'tutorial penggunaan', 'tolong', 'bantu', 'saya tidak mengerti', 'saya tidak paham', 'ga ngerti', 'gimana caranya', 'cara penggunaan', 'mau tanya', 'bisa tanya apa', 'fitur apa saja', 'help', 'assist', 'saya bingung', 'bingung', 'petunjuk', 'instruksi', 'cara bertanya', 'gimana nanya', 'bantu saya', 'aku tidak paham fungsi nya']
    if any(kw in user_input_normalized for kw in help_keywords):
        print(f"🎯 NEW ATURAN #4 (Help/Bantuan): Explicit help keyword detected. Triggering 'bantuan' intent.")
        bantuan_intent = next((i for i in INTENTS if i['name'] == 'bantuan'), None)
        if bantuan_intent:
            return format_response(bantuan_intent['responses'][0])

    # ===== NEW ATURAN #5 (DAFTAR PROYEK) - Prioritas tinggi, setelah fungsional inti dan sebelum welcome =====
    strong_daftar_proyek_keywords = [
        'daftar proyek', 'proyek apa saja', 'list proyek', 'semua proyek',
        'perumahan apa yang ada', 'pilihan proyek', 'proyek yang tersedia',
        'daftar rumah', 'berikan saya daftar rumah', 'daftar perumahan', 
        'berikan saya daftar perumahan', 'apa saja proyek kianoland', 
        'properti apa yang tersedia', 'list perumahan', 'tampilkan proyek', 
        'pilihan rumah', 'semua proyek', 'katalog proyek', 'rumah apa saja yang dijual', 
        'proyek yang masih tersedia', 'project yang ada di kianoland',
        'saya ingin lihat proyek yang ada', 'saya ingin lihat lihat', 'saya ingin lihat project', 
        'saya ingin lihat rumah yang ada', 'ada rumah apa aja',
        'berikan saya list project nya', 'berikan saya list proyek nya', 
        'saya mau lihat', 'saya mau lihat rumah', 'mau lihat rumah', 'lihat properti', 
        'nama proyeknya apa', # Tambahkan ini untuk mencocokkan "Nama proyeknya apa?"
        'produk apa yang dijual oleh kianoland group', # Tambahkan ini untuk mencocokkan "Produk apa yang dijual oleh Kianoland Group?"
        'produk kianoland group', 'apa saja produk kianoland', 'properti di kianoland group',
        'kasih lihat dong pilihan rumahnya', 'ada pilihan properti apa aja', 
        'mau cek proyek yang ready', 'saya mau tahu produknya', # Tambahkan ini untuk mencocokkan "Saya mau tahu produknya"
        'kalian jual rumah apa aja',
        'info properti', 'informasi properti', # Info eksplisit untuk properti
        'info', 'informasi', 'rumah', 'properti', 'perumahan', # Kata kunci info umum yang mungkin masuk ke sini
        'tampilkan proyek', 'lihat daftar', 'lihat proyek', # Menambahkan frasa "lihat" yang lebih umum
        'saya ingin lihat-lihat', # Menambahkan frasa gagal yang persis lagi agar lebih kuat
        'nama proyek', 'produk apa' # Menambahkan untuk "Nama proyeknya apa?" dan "Produk apa yang dijual..."
    ]

    # Pastikan bagian ini berada di atas ATURAN #6 (Welcome Intent)
    for keyword in strong_daftar_proyek_keywords:
        # Menggunakan re.search dengan word boundary untuk pencocokan yang lebih tepat
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input_normalized):
            print(f"🎯 NEW ATURAN #5 (General List): Strong keyword '{keyword}' for 'daftar_proyek' detected. Triggering 'daftar_proyek' intent.")
            daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
            if daftar_intent:
                return format_response(daftar_intent['responses'][0])

    # ===== NEW ATURAN #6 (Welcome/Greeting) - Paling bawah setelah semua intent fungsional =====
    # Pastikan ini berada di bawah ATURAN #5
    welcome_keywords_refined = [
        'halo', 'hi', 'hai', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam',
        'assalamualaikum', 'permisi', 'p', 'pe', 'mulai', '/mulai', 'start', '/start',
        'apa kabar', 'hai bot', 'hello kianoland', 'awali chat', 'bagaimana hari ini', 
        'bot', 'kianoland bot', 'kianoland group', 'saya baru di sini', 'perkenalkan diri',
        'siapa anda', 'ada yang bisa saya bantu', 'apakah ada yang bisa saya bantu hari ini'
    ] 
    if any(kw in user_input_normalized for kw in welcome_keywords_refined): # Gunakan daftar yang sudah disaring
        print(f"🎯 NEW ATURAN #6 (Welcome): Greeting keyword detected. Triggering 'welcome' intent.")
        welcome_intent = next((i for i in INTENTS if i['name'] == 'welcome'), None)
        if welcome_intent:
            return format_response(welcome_intent['responses'][0])

    # ===== ATURAN #7: Prioritaskan pertanyaan yang mengandung nama proyek =====
    if project:
        # ===== ATURAN #7A: TANGANI PROYEK YANG TIDAK ADA SAMA SEKALI (contoh: Kiano 4) =====
        if not is_valid_project(project):
            print(f"🎯 ATURAN #7A: Unknown project '{project}' detected.")
            return format_response(
                f"Maaf, proyek '{project}' tidak ada atau tidak tersedia di Kianoland Group.\n\n"
                f"Proyek yang tersedia saat ini:\n• Natureland Kiano 3\n• Green Jonggol Village"
            )
            
        # ===== ATURAN #7B: TANGANI PROYEK YANG ADA TAPI SUDAH SOLD OUT (contoh: Kiano 1) =====
        sold_out_projects = ["Natureland Kiano 1", "Natureland Kiano 2"]
        is_asking_specific_info = any(
            kw in user_input_normalized for kw in ['lokasi', 'alamat', 'peta', 'letak', 'harga', 'cicilan', 'promo', 'fasilitas', 'syarat']
        )
        if project in sold_out_projects and not is_asking_specific_info:
            print(f"🎯 ATURAN #7B: Sold Out Project '{project}' detected and no specific info requested.")
            return format_response(
                f"Maaf, proyek {project} sudah sold out. Kami merekomendasikan proyek terbaru kami:\n\n"
                f"🏡 Natureland Kiano 3 (Cibarusah, Bekasi)\n🌳 Green Jonggol Village (Jonggol, Bogor)\n\n"
                f"Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
            )
            
        # ===== ATURAN #7C: PERTANYAAN SPESIFIK BERDASARKAN KATA KUNCI (dengan proyek terdeteksi) =====
        specific_keywords_map = {
            'info_promo': ['promo', 'diskon', 'dp', 'uang muka'],
            'info_harga': ['harga', 'cicilan', 'angsuran', 'biaya', 'pl', 'pricelist'],
            'info_fasilitas': ['fasilitas'],
            'info_lokasi': ['lokasi', 'alamat', 'peta', 'letak'],
        }

        # Check for specific project info (including types like subsidi/komersil) FIRST
        if project == 'Natureland Kiano 3' and tipe_kiano3:
            print(f"🎯 ATURAN #7C (Specific Kiano 3 Type Info): Project '{project}' and Type '{tipe_kiano3}' Detected.")
            info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
            if info_intent:
                response_text = process_conditional_templates(info_intent['responses'][0], project=project, primary=tipe_kiano3)
                return format_response(response_text)
        elif project == 'Green Jonggol Village' and tipe_gjv: # NEW: Handle GJV specific types
            print(f"🎯 ATURAN #7C (Specific GJV Type Info): Project '{project}' and Type '{tipe_gjv}' Detected.")
            info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
            if info_intent:
                response_text = process_conditional_templates(info_intent['responses'][0], project=project, primary=tipe_gjv)
                return format_response(response_text)
        
        # Now, proceed with other specific keywords
        for intent_name, keywords in specific_keywords_map.items():
            if any(kw in user_input_normalized for kw in keywords):
                print(f"🎯 ATURAN #7C: Specific Intent '{intent_name}' with Project '{project}' Detected.")
                
                # Special handling for info_harga based on project and specific types
                if intent_name == 'info_harga':
                    primary_key = None
                    if project == 'Green Jonggol Village':
                        if tipe_gjv: 
                            primary_key = tipe_gjv
                        elif 'subsidi' in user_input_normalized: primary_key = 'GJV_subsidi'
                        elif 'komersil' in user_input_normalized: primary_key = 'GJV_komersil'
                        
                        if tipe_rumah and not primary_key:
                            if tipe_rumah == '30/60': primary_key = 'GJV_subsidi'
                            elif tipe_rumah == '36/72': primary_key = 'GJV_komersil'
                            else:
                                return format_response(f"Maaf, tipe rumah {tipe_rumah} tidak tersedia di Green Jonggol Village.\nTipe yang tersedia: 30/60 (Subsidi) & 36/72 (Komersil).")

                    elif project == 'Natureland Kiano 3':
                        if tipe_kiano3: 
                            primary_key = tipe_kiano3
                        elif '40/60' in user_input_normalized or '1 lantai' in user_input_normalized: primary_key = 'K3_1_Lantai'
                        elif '60/60' in user_input_normalized or 'mezzanine' in user_input_normalized or '1.5 lantai' in user_input_normalized: primary_key = 'K3_Mezzanine'
                        elif '90/60' in user_input_normalized or '2 lantai' in user_input_normalized: primary_key = 'K3_2_Lantai'
                    
                    forced_intent = next((i for i in INTENTS if i['name'] == 'info_harga'), None)
                    if forced_intent:
                        response_text = process_conditional_templates(forced_intent['responses'][0], project=project, primary=primary_key)
                        return format_response(response_text)
                
                # Logika umum untuk intent spesifik lainnya (dengan proyek)
                if intent_name != 'info_proyek': 
                    forced_intent = next((i for i in INTENTS if i['name'] == intent_name), None)
                    if forced_intent:
                        response_text = process_conditional_templates(forced_intent['responses'][0], project, lokasi)
                        return format_response(response_text)

        # ===== ATURAN #7D: INFO PROYEK VALID (catch-all for "info [project]" or just "[project]") =====
        general_info_keywords = ['info', 'informasi', 'detail', 'tentang', 'apa itu', 'apakah', 'ada']
        if project and (
            any(kw in user_input_normalized for kw in general_info_keywords) or 
            not any(kw in user_input_normalized for kw in sum(specific_keywords_map.values(), [])) 
        ):
            print(f"🎯 ATURAN #7D: General Info Request for Valid Project '{project}'.")
            info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
            if info_intent:
                response_text = process_conditional_templates(info_intent['responses'][0], project=project, primary=project)
                return format_response(response_text)


    # ===== ATURAN FALLBACK (Jika tidak ada proyek spesifik yang terdeteksi) =====

    # ===== ATURAN #8: General Info Harga (tanpa proyek spesifik) =====
    general_harga_keywords = ['harga', 'cicilan', 'angsuran', 'biaya', 'pl', 'pricelist']
    if any(kw in user_input_normalized for kw in general_harga_keywords) and not project:
        print("🎯 ATURAN #8: General Price/Pricelist Request Detected (no project).")
        return format_response(
            "Untuk proyek mana Anda ingin melihat pricelist?\n"
            "Misal: 'harga Natureland Kiano 3' atau 'pricelist Green Jonggol Village'."
        )

    # ===== ATURAN #9: General Info Lokasi (tanpa proyek spesifik) =====
    general_lokasi_keywords = ['lokasi', 'alamat', 'peta', 'letak', 'dimana', 'lihat lokasi']
    # Pengecualian: jika ada "kantor" (tanpa "alamat") maka biarkan jatuh ke info_kontak
    if any(kw in user_input_normalized for kw in general_lokasi_keywords):
        if 'kantor' in user_input_normalized and 'alamat' not in user_input_normalized:
            pass # Biarkan jatuh ke info_kontak jika hanya "kantor" tanpa "alamat"
        else:
            print(f"🎯 ATURAN #9: General Location Request Detected (no project).")
            return format_response(
                "Tentu, lokasi untuk proyek mana yang ingin Anda ketahui?\n\n"
                "Proyek yang tersedia:\n"
                "• Natureland Kiano 3\n"
                "• Green Jonggol Village"
            )
    
    # ===== ATURAN #10: General Info Fasilitas (tanpa proyek spesifik) =====
    general_fasilitas_keywords = ['fasilitas', 'fasilitasnya apa', 'apa fasilitasnya']
    if any(kw in user_input_normalized for kw in general_fasilitas_keywords) and not project:
        print(f"🎯 ATURAN #10: General Facility Request Detected (no project).")
        return format_response(
            "Tentu, informasi fasilitas untuk proyek mana yang ingin Anda ketahui?\n\n"
            "Proyek yang tersedia:\n"
            "• Natureland Kiano 3\n"
            "• Green Jonggol Village"
        )

    # ===== ATURAN #11: General Promo Request (if no project mentioned) =====
    general_promo_keywords = ['promo', 'diskon', 'dp', 'uang muka']
    if any(kw in user_input_normalized for kw in general_promo_keywords) and not project:
        print(f"🎯 ATURAN #11: General Promo Request Detected (no project).")
        promo_intent = next((i for i in INTENTS if i['name'] == 'info_promo'), None)
        if promo_intent:
            response_text = process_conditional_templates(promo_intent['responses'][0], project='all_promos')
            return format_response(response_text)


    # ===== ATURAN #12: RUMAH SUBSIDI & KOMERSIL (if no specific project was given) =====
    if ('subsidi' in user_input_normalized or 'komersil' in user_input_normalized) and not project:
        project_for_subsidi_komersil = "Green Jonggol Village"
        info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
        if info_intent:
            intro_text = "Untuk rumah subsidi, kami merekomendasikan **Green Jonggol Village**.\n\nBerikut informasinya:\n" if 'subsidi' in user_input_normalized else "Untuk rumah komersil, kami merekomendasikan **Green Jonggol Village**.\n\nBerikut informasinya:\n"
            primary_key_for_gjv = 'GJV_subsidi' if 'subsidi' in user_input_normalized else 'GJV_komersil'
            processed_response = process_conditional_templates(info_intent['responses'][0], project=project_for_subsidi_komersil, primary=primary_key_for_gjv)
            return format_response(intro_text + processed_response)


    # ===== ATURAN #13: REKOMENDASI LOKASI =====
    rekomendasi_keywords = ['rekomendasi', 'rekom', 'sarankan', 'saran', 'cocok', 'hunian', 'cari', 'ada apa', 'pilihan']
    if lokasi and any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print(f"🎯 ATURAN #13A: Recommendation for Known Location '{lokasi}' detected.")
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None) 
        if rekomendasi_intent:
            response_text = process_conditional_templates(rekomendasi_intent['responses'][0], lokasi=lokasi)
            return format_response(response_text)
    elif any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print("🎯 ATURAN #13B: General Recommendation Request (no location). Triggering 'daftar_proyek' intent.")
        daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
        if daftar_intent:
            return format_response(daftar_intent['responses'][0])
            
    # ===== ATURAN #14: PENCOCOKAN KEMIRIPAN UMUM (FALLBACK jika tidak ada yang lebih spesifik) =====
    print(f"🚦 Proceeding to Rule #14: Similarity-based matching. User input: '{user_input_normalized}'")
    best_match = None
    highest_score = 0.75 
    for intent in INTENTS:
        # Exclude already handled high-priority intents from similarity matching
        if intent['name'] in [
            'default_fallback', 'info_promo', 'info_harga', 'info_lokasi', 'info_fasilitas',
            'syarat_dokumen', 'rekomendasi_proyek', 'minat_beli', 'info_kontak', 'bantuan', 'daftar_proyek', 'welcome'
            ]: 
            continue
        
        for phrase in intent.get('phrases', []):
            similarity = SequenceMatcher(None, user_input_normalized, phrase).ratio()
            if similarity > highest_score:
                highest_score = similarity
                best_match = intent
                print(f"   - Found better similarity match: {best_match['name']} ({highest_score:.2f}) with phrase: '{phrase}'")
    
    if best_match:
        print(f"🎯 Best match by similarity: {best_match['name']} (score: {highest_score:.2f})")
        response_text = process_conditional_templates(best_match['responses'][0], project, lokasi)
        return format_response(response_text)


    # ===== ATURAN #15: FALLBACK TERAKHIR (If no other rule fires) =====
    print("🛑 Final Fallback.")
    fallback_intent = next((i for i in INTENTS if i['name'] == 'default_fallback'), None)
    if fallback_intent:
        return format_response(fallback_intent['responses'][0])
    return format_response("Maaf, saya tidak dapat memproses permintaan Anda saat ini.")

def process_conditional_templates(text: str, project: str = None, lokasi: str = None, primary: str = None, secondary: str = None) -> str:
    """Process conditional templates with intelligent block selection based on project or location."""

    # Prioritize 'primary' selector, then 'project', then 'lokasi'
    selector_to_use = primary or project or lokasi

    if selector_to_use:
        escaped_selector = re.escape(selector_to_use)
        # Match the specific block using the determined selector
        pattern = r'\{\{#' + escaped_selector + r'\}\}(.*?)\{\{/' + escaped_selector + r'\}\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Remove any other conditional blocks within the selected content
            content = re.sub(r'\{\{#[^}]+\}\}', '', content)
            content = re.sub(r'\{\{/[^}]+\}\}', '', content)
            # Replace remaining placeholders (like {{proyek}}, {{lokasi}} if any)
            content = content.replace("{{proyek}}", project if project else "")
            content = content.replace("{{lokasi}}", lokasi if lokasi else "")
            return content

    # If no specific block is found, try to find a general 'all_promos' block or 'fallback'
    if 'all_promos' in text: # Special case for general promo
        all_promos_pattern = r'\{\{#all_promos\}\}(.*?)\{\{/all_promos\}\}'
        all_promos_match = re.search(all_promos_pattern, text, re.DOTALL)
        if all_promos_match:
            return all_promos_match.group(1).strip()

    fallback_pattern = r'\{\{#fallback\}\}(.*?)\{\{/fallback\}\}'
    fallback_match = re.search(fallback_pattern, text, re.DOTALL)
    if fallback_match:
        fallback_text = fallback_match.group(1).strip()
        fallback_text = fallback_text.replace("{{proyek}}", project if project else "")
        fallback_text = fallback_text.replace("{{lokasi}}", lokasi if lokasi else "")
        return fallback_text

    # If no specific or fallback block, remove all conditional tags
    text = re.sub(r'\{\{#[^}]+\}\}', '', text)
    text = re.sub(r'\{\{/[^}]+\}\}', '', text)
    return text.strip()


def format_response(text: str) -> Dict[str, str]:
    """Format response for all platforms"""
    # Ganti nomor telepon generik dengan nomor yang sudah diseragamkan
    text_with_contact = text.replace('{{kontak_marketing}}', marketing_contacts['default'])
    text_with_contact = text_with_contact.replace('\\n', '\n').replace('\\"', '"')

    return {
        'raw': text_with_contact,
        'discord': text.replace('bold_start', '**').replace('bold_end', '**'),
        'telegram': text.replace('**', '').replace('bold_start', '<b>').replace('bold_end', '</b>'),
        'web': text.replace('**', '').replace('bold_start', '<strong>').replace('bold_end', '</strong>')
    }

# Initialize on import
load_resources()