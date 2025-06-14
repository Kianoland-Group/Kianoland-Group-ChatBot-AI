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
    """Detects projects, locations, and house types from user input."""
    detected = {}
    text_lower = text.lower()

    # Removed project_stop_words from here to avoid interference with intent matching

    # --- Deteksi Proyek ---
    for entry in ENTITIES.get('proyek', []):
        for synonym in entry.get('synonyms', []):
            # Use word boundaries (\b) to match whole words only
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                # Ensure the detected synonym is not a generic "property" or "proyek"
                # This check becomes more important without a global stop_words list here
                if synonym.lower() not in ['proyek', 'project', 'properti', 'rumah', 'perumahan']:
                    detected['proyek'] = entry['value']
                    print(f"✅ Proyek terdeteksi (from entity): '{synonym}' -> {entry['value']}")
                    break
        if 'proyek' in detected:
            break
            
    if 'proyek' not in detected:
        match = re.search(r'\b(kiano|nlk)\s*(\d+)\b', text_lower)
        if match:
            project_base_name = "Natureland Kiano"
            project_number = match.group(2)
            constructed_name = f"{project_base_name} {project_number}"
            # Only add if it's not a general query like "kiano berapa"
            if constructed_name.lower() not in ['natureland kiano group']: # Add more exclusions if needed
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

    # --- PERBAIKAN BARU: Deteksi Tipe Rumah ---
    for entry in ENTITIES.get('tipe_rumah', []):
        for synonym in entry.get('synonyms', []):
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, text_lower):
                detected['tipe_rumah'] = entry['value']
                print(f"✅ Tipe Rumah terdeteksi: '{synonym}' -> {entry['value']}")
                break
        if 'tipe_rumah' in detected:
            break

    # --- PERBAIKAN BARU: Deteksi Tipe Rumah Kiano 3 (Manual) ---
    kiano3_types = {
        'K3_1_Lantai': ['1 lantai', 'satu lantai', '40/60'],
        'K3_Mezzanine': ['mezzanine', '1,5 lantai', 'satu setengah lantai', '60/60'],
        'K3_2_Lantai': ['2 lantai', 'dua lantai', '90/60']
    }
    # Lakukan perulangan untuk setiap tipe
    for key, synonyms in kiano3_types.items():
        # Lakukan perulangan untuk setiap sinonim
        for synonym in synonyms:
            # Cek apakah sinonim ada di dalam kalimat pengguna
            if synonym in text_lower:
                # Jika ditemukan, simpan entitasnya dengan nama internal (key)
                detected['tipe_kiano3'] = key
                print(f"✅ Tipe Kiano 3 terdeteksi (manual): '{synonym}' -> {key}")
                # Hentikan pencarian sinonim untuk tipe ini
                break 
        
        # Jika entitas sudah ditemukan, hentikan pencarian tipe lainnya
        if 'tipe_kiano3' in detected:
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

    # ===== NEW ATTEMPT - ATURAN #0: Prioritaskan 'daftar_proyek' with strong keywords =====
    # Define a more specific set of keywords for 'daftar_proyek' that are less likely to be generic stop words
    # Added "available properties", "list of houses", "what projects are there"
    # ENHANCED: Added more common phrases and variations.
    strong_daftar_proyek_keywords = [
        'daftar proyek', 'proyek apa saja', 'list proyek', 'semua proyek',
        'perumahan apa yang ada', 'pilihan proyek', 'proyek yang tersedia',
        'daftar rumah', 'daftar perumahan', 
        'properti apa saja', 'property apa yang tersedia', 
        'list perumahan', 'tampilkan proyek', 'pilihan rumah', 'katalog proyek',
        'rumah apa saja yang dijual', 'proyek yang masih tersedia', 'project yang ada di kianoland',
        'ingin lihat proyek', 'lihat project', 'lihat rumah yang ada', 'ada rumah apa aja',
        'list proyek nya', 'mau lihat rumah', 'lihat properti', 
        'proyek kianoland', 'informasi properti kianoland', 'properti kianoland group',
        'sebutkan property', 'sebutkan proyek', 'properti apa', 'rumah apa',
        'property apa aja', 'berikan list property', 'property apa aja yang ada di kiano',
        'properti yang ada', 'property yang ada', 'kianoland property', 'list properti kianoland',
        'daftar properti kianoland', 'info', 'informasi', 'berikan saya info',
        'info properti', 'saya ingin lihat info', 'rumah yang tersedia', 'project yang ada',
        'daftar hunian', 'hunian apa saja', 'list hunian', 'perumahan', 'rumah',
        'list', 'daftar', 'apa aja' # Added more generic but relevant keywords
    ]

    # Menggunakan regex untuk pencocokan kata utuh agar lebih robust
    # Ini akan mencari salah satu keyword dalam user_input_normalized
    for keyword in strong_daftar_proyek_keywords:
        # Use re.search with word boundaries (\b) for full word match,
        # but also allow partial match for short keywords like "info" if it's the whole input
        if user_input_normalized == keyword: # Exact match for any keyword (single or multi-word)
            print(f"🎯 ATURAN #0 (NEW - Exact Match): Full input '{keyword}' for 'daftar_proyek' detected. Triggering 'daftar_proyek' intent.")
            daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
            if daftar_intent:
                return format_response(daftar_intent['responses'][0]) # <--- THIS RETURN IS CRUCIAL
        elif re.search(r'\b' + re.escape(keyword) + r'\b', user_input_normalized):
            print(f"🎯 ATURAN #0 (NEW - Keyword Match): Strong keyword '{keyword}' for 'daftar_proyek' detected. Triggering 'daftar_proyek' intent.")
            daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
            if daftar_intent:
                return format_response(daftar_intent['responses'][0]) # <--- THIS RETURN IS CRUCIAL

    # Handle Discord-specific !info command explicitly at the beginning if needed
    if user_input_normalized == '!info':
        print(f"🎯 ATURAN #0 (Discord Command): '!info' detected. Triggering 'daftar_proyek' intent.")
        daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
        if daftar_intent:
            return format_response(daftar_intent['responses'][0]) # <--- THIS RETURN IS CRUCIAL


    # ===== ATURAN #1A: TANGANI PROYEK YANG TIDAK ADA SAMA SEKALI (contoh: Kiano 4) =====
    if project and not is_valid_project(project):
        print(f"🎯 ATURAN #1A: Unknown project '{project}' detected.")
        return format_response( # <--- This will return and exit
            f"Maaf, proyek '{project}' tidak ada atau tidak tersedia di Kianoland Group.\n\n"
            f"Proyek yang tersedia saat ini:\n• Natureland Kiano 3\n• Green Jonggol Village"
        )

    # ===== ATURAN #1B: TANGANI PROYEK YANG ADA TAPI SUDAH SOLD OUT (contoh: Kiano 1) =====
    sold_out_projects = ["Natureland Kiano 1", "Natureland Kiano 2"]
    # Check if a project is mentioned AND it's a sold-out project.
    # Also, ensure it's not a query specifically asking for its location, as that might still be relevant.
    is_asking_lokasi = any(kw in user_input_normalized for kw in ['lokasi', 'alamat', 'peta', 'letak'])
    if project and project in sold_out_projects and not is_asking_lokasi:
        print(f"🎯 ATURAN #1B: Sold Out Project '{project}' detected.")
        return format_response( # <--- This will return and exit
            f"Maaf, proyek {project} sudah sold out. Kami merekomendasikan proyek terbaru kami:\n\n"
            f"🏡 Natureland Kiano 3 (Cibarusah, Bekasi)\n🌳 Green Jonggol Village (Jonggol, Bogor)\n\n"
            f"Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
        )


    # ===== ATURAN #2: PERTANYAAN SPESIFIK BERDASARKAN KATA KUNCI =====
    specific_keywords_map = {
        'info_promo': ['promo', 'diskon', 'dp', 'uang muka'],
        'info_harga': ['harga', 'cicilan', 'angsuran', 'biaya', 'pl', 'pricelist'],
        'info_fasilitas': ['fasilitas'],
        'info_lokasi': ['lokasi', 'alamat', 'peta', 'letak'],
        'syarat_dokumen': ['syarat', 'persyaratan', 'dokumen', 'kpr'],
        'bantuan': ['bantuan', 'panduan', 'tolong', 'bantu', 'tidak mengerti', 'tidak paham', 'bingung'],
        'minat_beli': ['beli', 'minat', 'lihat']
    }
    
    for intent_name, keywords in specific_keywords_map.items():
        if any(kw in user_input_normalized for kw in keywords):
            
            if intent_name == 'info_promo' and not project:
                # ... (Logika promo umum)
                print("🎯 ATURAN #2.A: General Promo Request Detected.")
                promo_intent = next((i for i in INTENTS if i['name'] == 'info_promo'), None)
                if promo_intent:
                    # Pass a special keyword 'all_promos' to process_conditional_templates
                    response_text = process_conditional_templates(promo_intent['responses'][0], project='all_promos')
                    return format_response(response_text) # <--- This will return and exit

            # --- PERBAIKAN FINAL: Logika Cerdas untuk Harga berdasarkan Proyek & Tipe Rumah ---
            if intent_name == 'info_harga':
                primary_key = None
                
                if project == 'Green Jonggol Village':
                    print("🎯 ATURAN #2.B: Specific GJV Price Request Detected.")
                    if tipe_rumah == '30/60': primary_key = 'GJV_subsidi'
                    elif tipe_rumah == '36/72': primary_key = 'GJV_komersil'
                    elif 'subsidi' in user_input_normalized: primary_key = 'GJV_subsidi'
                    elif 'komersil' in user_input_normalized: primary_key = 'GJV_komersil'
                    
                    if tipe_rumah and not primary_key:
                        return format_response(f"Maaf, tipe rumah {tipe_rumah} tidak tersedia di Green Jonggol Village.\nTipe yang tersedia: 30/60 (Subsidi) & 36/72 (Komersil).") # <--- This will return and exit

                elif project == 'Natureland Kiano 3':
                    print("🎯 ATURAN #2.B: Specific Kiano 3 Price Request Detected.")
                    if tipe_kiano3 == 'K3_1_Lantai': primary_key = 'K3_1_Lantai'
                    elif tipe_kiano3 == 'K3_Mezzanine': primary_key = 'K3_Mezzanine'
                    elif tipe_kiano3 == 'K3_2_Lantai': primary_key = 'K3_2_Lantai'
                
                forced_intent = next((i for i in INTENTS if i['name'] == 'info_harga'), None)
                if forced_intent:
                    response_text = process_conditional_templates(forced_intent['responses'][0], project=project, primary=primary_key)
                    return format_response(response_text) # <--- This will return and exit
            
            # --- Logika umum untuk intent spesifik lainnya ---
            print(f"🎯 ATURAN #2.C: General Specific Intent '{intent_name}' Detected.")
            forced_intent = next((i for i in INTENTS if i['name'] == intent_name), None)
            if forced_intent:
                response_text = process_conditional_templates(forced_intent['responses'][0], project, lokasi)
                return format_response(response_text) # <--- This will return and exit
    
    # ===== ATURAN #2B.5 (BARU): INFO SPESIFIK TIPE RUMAH KIANO 3 =====
    # Aturan ini menangani pertanyaan seperti "info rumah 1 lantai di kiano 3"
    if project == 'Natureland Kiano 3' and tipe_kiano3:
        print(f"🎯 ATURAN #2B.5: Specific info for Kiano 3 type '{tipe_kiano3}'")
        info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
        if info_intent:
            # Gunakan 'tipe_kiano3' sebagai kunci untuk memilih blok respons yang benar
            response_text = process_conditional_templates(info_intent['responses'][0], project='Natureland Kiano 3', primary=tipe_kiano3)
            return format_response(response_text) # <--- This will return and exit

    # ===== ATURAN #2C: INFO PROYEK VALID =====
    if project and project not in sold_out_projects:
        print(f"🎯 ATURAN #2C: General Info Request for Valid Project '{project}'.")
        info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
        if info_intent:
            response_text = process_conditional_templates(info_intent['responses'][0], project)
            return format_response(response_text) # <--- This will return and exit

    # ===== ATURAN #3: RUMAH SUBSIDI & KOMERSIL =====
    if 'subsidi' in user_input_normalized or 'komersil' in user_input_normalized:
        project = "Green Jonggol Village"
        info_intent = next((i for i in INTENTS if i['name'] == 'info_proyek'), None)
        if info_intent:
            intro_text = "Untuk rumah subsidi, kami merekomendasikan **Green Jonggol Village**.\n\nBerikut informasinya:\n" if 'subsidi' in user_input_normalized else "Untuk rumah komersil, kami merekomendasikan **Green Jonggol Village**.\n\nBerikut informasinya:\n"
            processed_response = process_conditional_templates(info_intent['responses'][0], project=project)
            return format_response(intro_text + processed_response) # <--- This will return and exit

    # ===== ATURAN #4: REKOMENDASI LOKASI =====
    rekomendasi_keywords = ['rekomendasi', 'rekom', 'sarankan', 'saran', 'cocok', 'hunian'] # Removed 'rumah', 'proyek', 'properti' to prevent false positives with daftar_proyek
    if lokasi and any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print(f"🎯 ATURAN #4A: Recommendation for Known Location '{lokasi}' detected.")
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None)
        if rekomendasi_intent:
            response_text = process_conditional_templates(rekomendasi_intent['responses'][0], lokasi=lokasi)
            return format_response(response_text) # <--- This will return and exit
    elif not lokasi and any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print("🎯 ATURAN #4B: Recommendation for Unknown Location detected.")
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None)
        if rekomendasi_intent:
            # Ensure 'lokasi' is explicitly passed as None or empty to prevent incorrect block matching
            response_text = process_conditional_templates(rekomendasi_intent['responses'][0], lokasi=None) # Changed to None
            return format_response(response_text) # <--- This will return and exit

    # ===== ATURAN #5: PENCOCOKAN KEMIRIPAN UMUM (FALLBACK) =====
    print("🚦 Proceeding to Rule #5: Similarity-based matching.")
    best_match = None
    highest_score = 0.75
    for intent in INTENTS:
        # Exclude common and specific intents that should be handled by earlier rules
        if intent['name'] in ['default_fallback', 'info_promo', 'info_harga', 'info_fasilitas', 'info_lokasi', 'syarat_dokumen', 'bantuan', 'rekomendasi_proyek', 'daftar_proyek']:
            continue
        for phrase in intent.get('phrases', []):
            similarity = SequenceMatcher(None, user_input_normalized, phrase).ratio()
            if similarity > highest_score:
                highest_score = similarity
                best_match = intent
    if best_match:
        print(f"🎯 Best match by similarity: {best_match['name']} (score: {highest_score:.2f})")
        response_text = process_conditional_templates(best_match['responses'][0], project)
        return format_response(response_text) # <--- This will return and exit
    
    # ===== ATURAN #6: FALLBACK TERAKHIR =====
    print("🛑 Final Fallback.")
    fallback_intent = next((i for i in INTENTS if i['name'] == 'default_fallback'), None)
    if fallback_intent:
        return format_response(fallback_intent['responses'][0]) # <--- This will return and exit
    return format_response("Maaf, saya tidak dapat memproses permintaan Anda saat ini.")

def process_conditional_templates(text: str, project: str = None, lokasi: str = None, primary: str = None, secondary: str = None) -> str:
    """Process conditional templates with intelligent block selection based on project or location."""

    # --- Tentukan kunci selektor utama (bisa primary, project, atau lokasi) ---
    selector = primary or project or lokasi

    # 1. Coba cari blok spesifik menggunakan selektor
    if selector:
        escaped_selector = re.escape(selector)
        # Ensure that only the exact selector block is matched
        pattern = r'\{\{#' + escaped_selector + r'\}\}(.*?)\{\{/' + escaped_selector + r'\}\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Replace placeholder in the selected content
            content = content.replace("{{proyek}}", project if project else "")
            content = content.replace("{{lokasi}}", lokasi if lokasi else "")
            return content

    # 2. Jika tidak ada blok spesifik, coba cari blok fallback
    # THIS IS THE CRITICAL PART: The previous `fallback` in `rekomendasi_proyek`
    # was triggered when `lokasi` was None or empty, leading to "Maaf, kami belum memiliki proyek di lokasi tersebut."
    # We need to ensure that the general fallback is used if no specific project/location block is found,
    # OR if the original text does not contain any of the conditional blocks.
    fallback_pattern = r'\{\{#fallback\}\}(.*?)\{\{/fallback\}\}'
    fallback_match = re.search(fallback_pattern, text, re.DOTALL)
    if fallback_match:
        fallback_text = fallback_match.group(1).strip()
        # Replace placeholder in the fallback text
        fallback_text = fallback_text.replace("{{proyek}}", project if project else "")
        fallback_text = fallback_text.replace("{{lokasi}}", lokasi if lokasi else "")
        return fallback_text

    # 3. If no matching block (specific or fallback) is found, return the original text
    # after removing any remaining template tags. This should ideally not be reached
    # if all intents have proper fallbacks or specific responses.
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