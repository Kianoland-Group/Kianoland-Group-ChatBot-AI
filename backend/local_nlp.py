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
    """Detect entities using improved matching with substring and regex"""
    detected = {}
    text_lower = re.sub(r'[^\w\s]', '', text.lower())
    print(f"\n📥 Teks input entitas: '{text_lower}'")

    # Prioritize exact match with word boundaries
    for entity_type, entries in ENTITIES.items():
        for entry in entries:
            for synonym in entry.get('synonyms', []):
                norm_synonym = re.sub(r'\s+', ' ', synonym.lower()).strip()
                
                # Gunakan regex dengan word boundary
                pattern = r'\b' + re.escape(norm_synonym) + r'\b'
                if re.search(pattern, text_lower):
                    print(f"✅ Cocok (regex): '{synonym}' → {entry['value']}")
                    detected[entity_type] = entry['value']
                    break
            if entity_type in detected:
                break

    # Regex matching for all entities if not detected by substring
    for entity_type, entries in ENTITIES.items():
        if entity_type in detected:  # Skip if already detected
            continue
        for entry in entries:
            for synonym in entry.get('synonyms', []):
                escaped_synonym = re.sub(r'\s+', r'\\s+', re.escape(synonym.lower()))
                pattern = r'\b' + escaped_synonym + r'\b'
                if re.search(pattern, text_lower):
                    print(f"✅ Cocok (regex): '{synonym}' → {entry['value']}")
                    detected[entity_type] = entry['value']
                    break
            if entity_type in detected:
                break

    # Improved fallback for project detection
    if 'proyek' not in detected:
        text_lower = text.lower()
        
        # --- START OF FIX ---
        # Prioritaskan untuk mendeteksi pola "kiano [angka]" terlebih dahulu
        kiano_match = re.search(r'\bkiano\s*(\d+)\b', text_lower)
        if kiano_match:
            # Jika pola ditemukan, ambil angkanya
            num = kiano_match.group(1)
            detected['proyek'] = f"Natureland Kiano {num}"
        else:
            # Jika tidak ada pola "kiano [angka]", baru cari sinonim lain
            for entry in ENTITIES.get('proyek', []):
                # Cek value utama
                if entry['value'].lower() in text_lower:
                    detected['proyek'] = entry['value']
                    break
                # Cek sinonim lain
                for synonym in entry.get('synonyms', []):
                    if synonym.lower() in text_lower:
                        detected['proyek'] = entry['value']
                        break
                if 'proyek' in detected:
                    break
        # --- END OF FIX ---
        
        # FALLBACK BARU: Deteksi pola "info [nama_proyek]" jika belum terdeteksi
    if 'proyek' not in detected:
        match = re.search(r'info\s+([^\s]+(?:\s+[^\s]+){0,2})', text_lower)  # Max 3 words after "info"
        if match:
            project_candidate = match.group(1).strip()
            
            # Skip generic terms
            generic_terms = ['proyek', 'project', 'properti', 'rumah', 'perumahan', 'yang', 'ada', 'tersedia']
            if project_candidate.lower() not in generic_terms:
                # Check if it might be a list request
                if any(kw in project_candidate for kw in ['semua', 'list', 'daftar', 'apa saja', 'yang ada']):
                    # Don't treat as project name
                    print(f"⚠️ Skipped project candidate (list request): {project_candidate}")
                else:
                    detected['proyek'] = project_candidate
                    print(f"🎯 Detected project via fallback: {project_candidate}")

    # PERBAIKAN: Deteksi lokasi Jonggol secara eksplisit
    if 'lokasi' not in detected and ('jonggol' in text_lower or 'bogor' in text_lower or 'cileungsi' in text_lower):
        # Cek apakah sudah ada lokasi yang terdeteksi
        if not any(loc in detected.get('lokasi', '') for loc in ['Cibarusah', 'Jatisampurna', 'Bekasi', 'Cibubur']):
            detected['lokasi'] = "Jonggol"
            print(f"🎯 Detected location via fallback: Jonggol")

    print(f"🧩 Entitas terdeteksi: {detected}")
    return detected

def detect_intent_local(user_input: str) -> Dict[str, str]:
    """Detect intent using a final, robust, rule-based priority system."""
    user_input_normalized = re.sub(r'(\w)\1{2,}', r'\1', user_input.lower().strip())
    print(f"\n🔍 User input: '{user_input}' → Normalized: '{user_input_normalized}'")

    entities = detect_entities(user_input)
    project = entities.get('proyek')
    lokasi = entities.get('lokasi')

    # ===== ATURAN #1: VALIDASI PROYEK TIDAK VALID =====
    if project and not is_valid_project(project):
        print("🎯 ATURAN #1: Proyek Tidak Valid Terdeteksi.")
        projects_list = "\n".join(f"• {p}" for p in get_valid_projects())
        return format_response(
            f"Maaf, proyek '{project}' tidak tersedia di Kianoland Group.\n\n"
            f"Proyek yang tersedia saat ini:\n{projects_list}"
        )
    
    # ===== ATURAN #2: PERTANYAAN SPESIFIK (PROMO, HARGA, FASILITAS, LOKASI) =====
    specific_keywords = {
        'info_promo': ['promo', 'diskon', 'dp', 'uang muka'],
        'info_harga': ['harga', 'kpr', 'cicilan', 'angsuran', 'biaya'],
        'info_fasilitas': ['fasilitas'],
        'info_lokasi': ['lokasi', 'alamat', 'peta', 'letak', 'dimana']
    }
    for intent_name, keywords in specific_keywords.items():
        if any(kw in user_input_normalized for kw in keywords):
            
            # --- Logika Khusus untuk PROMO ---
            if intent_name == 'info_promo':
                latest_promo_keywords = ['terbaru', 'saat ini', 'berjalan', 'sekarang']
                # Jika pengguna menanyakan promo terkini -> tampilkan semua
                if any(kw in user_input_normalized for kw in latest_promo_keywords):
                    print("🎯 ATURAN #2.A: Info Semua Promo Terkini.")
                    promo_intent = next((i for i in INTENTS if i['name'] == 'info_promo'), None)
                    if promo_intent:
                        response_text = promo_intent['responses'][0]
                        # Gunakan kunci 'all_promos' untuk memanggil template yang benar
                        processed_response = process_conditional_templates(response_text, project='all_promos')
                        return format_response(processed_response)
            
            # Jika tidak ada nama proyek, minta klarifikasi (berlaku untuk semua)
            if not project:
                print(f"🎯 ATURAN #2.B: Klarifikasi untuk '{intent_name}'.")
                # Untuk 'promo', fallback-nya ada di dalam template. Untuk yg lain, kita buat manual.
                forced_intent = next((i for i in INTENTS if i['name'] == intent_name), None)
                if forced_intent:
                    response_text = forced_intent['responses'][0]
                    processed_response = process_conditional_templates(response_text, project=None)
                    return format_response(processed_response)

            # Jika ada nama proyek, paksa intent yang benar.
            print(f"🎯 ATURAN #2.C: Intent Spesifik '{intent_name}' Terdeteksi.")
            forced_intent = next((i for i in INTENTS if i['name'] == intent_name), None)
            if forced_intent:
                response_text = forced_intent['responses'][0]
                processed_response = process_conditional_templates(response_text, project, lokasi)
                return format_response(processed_response)

    # ===== ATURAN #3: REKOMENDASI LOKASI =====
    rekomendasi_keywords = ['rekomendasi', 'rekom', 'sarankan', 'saran', 'cocok', 'daerah', 'kawasan']
    if lokasi and any(kw in user_input_normalized for kw in rekomendasi_keywords):
        print("🎯 ATURAN #3: Rekomendasi Lokasi Terdeteksi.")
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None)
        if rekomendasi_intent:
            response_text = rekomendasi_intent['responses'][0]
            processed_response = process_conditional_templates(response_text, project=lokasi, lokasi=lokasi)
            return format_response(processed_response)

    # ===== ATURAN #4: FALLBACK KE PENCOCOKAN KEMIRIPAN =====
    print("🚦 Lanjut ke Aturan #4: Pencocokan Berbasis Kemiripan.")
    best_match = None
    highest_score = 0
    for intent in INTENTS:
        for phrase in intent.get('phrases', []):
            boost = 0.2 if project and intent['name'] == 'info_proyek' else 0
            similarity = SequenceMatcher(None, user_input_normalized, phrase).ratio() + boost
            if similarity > highest_score:
                highest_score = similarity
                best_match = intent
    
    required_score = 0.65
    if best_match and highest_score > required_score:
        print(f"🎯 Best match by similarity: {best_match['name']} (score: {highest_score:.2f})")
        response_text = best_match['responses'][0] if best_match['responses'] else ""
        processed_response = process_conditional_templates(response_text, project, lokasi)
        return format_response(processed_response)

    # ===== ATURAN #5: FALLBACK TERAKHIR =====
    print("🛑 Fallback Terakhir.")
    fallback_intent = next((i for i in INTENTS if i['name'] == 'default_fallback'), None)
    if fallback_intent:
        return format_response(fallback_intent['responses'][0])
    
    return format_response("Maaf, saya tidak memahami pertanyaan Anda. Ketik 'bantuan' untuk panduan.")

def process_conditional_templates(text: str, project: str = None, lokasi: str = None, primary: str = None, secondary: str = None) -> str:
    """Process conditional templates with intelligent block selection"""

        # If we have a primary key, try to find its specific block
    if primary:
        escaped_primary = re.escape(primary)
        pattern = r'\{\{#' + escaped_primary + r'\}\}(.*?)\{\{/' + escaped_primary + r'\}\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # PERBAIKAN: Coba gunakan secondary key jika primary tidak ditemukan
    if secondary and secondary != primary:
        escaped_secondary = re.escape(secondary)
        pattern = r'\{\{#' + escaped_secondary + r'\}\}(.*?)\{\{/' + escaped_secondary + r'\}\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # First replace the global placeholders
    if project:
        text = text.replace("{{proyek}}", project)
    else:
        text = text.replace("{{proyek}}", "")

    if lokasi:
        text = text.replace("{{lokasi}}", lokasi)
    else:
        text = text.replace("{{lokasi}}", "")

    # If we have a project, try to find its specific block
    if project:
        escaped_project = re.escape(project)
        pattern = r'\{\{#' + escaped_project + r'\}\}(.*?)\{\{/' + escaped_project + r'\}\}'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no specific block found, try to find fallback block
    # If no specific block found, try to find fallback block
    fallback_pattern = r'\{\{#fallback\}\}(.*?)\{\{/fallback\}\}'
    fallback_match = re.search(fallback_pattern, text, re.DOTALL)
    if fallback_match:
        fallback_text = fallback_match.group(1).strip()
        # Replace placeholder with actual project name
        if project and "{{proyek}}" in fallback_text:
            fallback_text = fallback_text.replace("{{proyek}}", project)
        return fallback_text
    
    # If no fallback, return the text with blocks removed
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