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
    """Mendapatkan daftar proyek valid yang tersedia (tanpa Kiano 1)"""
    available_projects = [
        "Natureland Kiano 2",
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
    """Detect intent from user input with improved entity handling"""
    # Normalize input
    user_input = re.sub(r'\s+', ' ', user_input).strip()
    user_input_normalized = re.sub(r'(\w)\1{2,}', r'\1', user_input.lower())
    print(f"\n🔍 User input: '{user_input}' → Normalized: '{user_input_normalized}'")

    # --- START PERBAIKAN ---
    # 1. Deteksi entitas di awal untuk semua kasus.
    entities = detect_entities(user_input)
    project = entities.get('proyek')

    # 2. Validasi proyek jika terdeteksi, sebelum memproses intent apapun.
    if project:
        # Handle kasus khusus untuk Kiano 1 (sold out)
        if "kiano 1" in project.lower():
            return format_response(
                "bold_startNatureland Kiano 1:bold_end\n"
                "Maaf, proyek ini sudah sold out. Kami merekomendasikan proyek terbaru kami:\n\n"
                "🏡 Natureland Kiano 2 (Jatisampurna, Bekasi)\n"
                "🏡 Natureland Kiano 3 (Cibarusah, Bekasi)\n"
                "🌳 Green Jonggol Village (Jonggol, Bogor)\n\n"
                "Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
            )
        
        # Handle proyek tidak valid lainnya (misal: Kiano 4)
        if not is_valid_project(project):
            projects_list = "\n".join(f"• {p}" for p in get_valid_projects())
            return format_response(
                f"Maaf, proyek '{project}' tidak tersedia di Kianoland Group.\n\n"
                f"Proyek yang tersedia:\n{projects_list}\n\n"
                f"Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
            )
    # --- AKHIR PERBAIKAN ---

    # --- PERBAIKAN 1: Prioritaskan Rekomendasi Berbasis Lokasi ---
    rekomendasi_keywords = ['rekomendasi', 'rekom', 'sarankan', 'saran', 'cocok', 'daerah', 'kawasan']
    lokasi = entities.get('lokasi')
    
    if any(kw in user_input_normalized for kw in rekomendasi_keywords) and lokasi:
        rekomendasi_intent = next((i for i in INTENTS if i['name'] == 'rekomendasi_proyek'), None)
        if rekomendasi_intent:
            response_text = rekomendasi_intent['responses'][0]
            # Gunakan lokasi sebagai parameter utama untuk memicu template yang benar
            processed_response = process_conditional_templates(response_text, lokasi, lokasi)
            return format_response(processed_response)
    # --- AKHIR PERBAIKAN 1 ---

    # ===== Logika Intent Lanjutan =====
    purchase_keywords = [
        'mau beli', 'ingin beli', 'minat beli', 'pesan rumah', 'booking rumah',
        'beli rumah', 'pembelian rumah', 'booking unit', 'lihat rumah', 'kunjung'
    ]
    # Jika intent adalah pembelian dan proyek sudah tervalidasi
    if any(phrase in user_input_normalized for phrase in purchase_keywords) and project:
        contact = marketing_contacts.get(project, "Bp. Toni - 0896 3823 0725")
        return format_response(
            f"Terima kasih atas minat Anda pada proyek {project}!\n\n"
            f"Untuk proses pembelian atau kunjungan proyek, silakan hubungi marketing kami:\n"
            f"📞 {contact}\n\n"
            "Atau kunjungi website resmi kami:\n"
            "🌐 https://kianolandgroup.com"
        )
    
    # Intent matching logic
    best_match = None
    highest_score = 0
    matched_phrase = ""

    for intent in INTENTS:
        for phrase in intent.get('phrases', []):
            phrase_normalized = phrase.lower()
            similarity = SequenceMatcher(None, user_input_normalized, phrase_normalized).ratio()
            
            # Apply keyword boosts only if base similarity is above threshold
            base_similarity = similarity
            if base_similarity > 0.4:  # Only apply boosts if already somewhat similar
                if intent['name'] == 'info_lokasi' and any(kw in user_input_normalized for kw in ['lokasi', 'alamat', 'letak', 'peta', 'dimana']):
                    similarity += 0.3
                elif intent['name'] == 'info_proyek' and any(kw in user_input_normalized for kw in ['info', 'informasi', 'detail', 'beli', 'rumah', 'lihat']):
                    similarity += 0.2
                elif intent['name'] == 'daftar_proyek' and any(kw in user_input_normalized for kw in ['daftar', 'list', 'semua', 'tersedia', 'ada', 'apa saja', 'yang ada']):
                    similarity += 0.3  # Only one boost
                elif intent['name'] == 'bantuan' and any(kw in user_input_normalized for kw in ['bantu', 'tolong', 'panduan', 'tidak mengerti', 'ga ngerti', 'help']):
                    similarity += 0.2
                elif intent['name'] == 'info_promo' and any(kw in user_input_normalized for kw in ['dp', 'promo', 'diskon', 'uang muka', 'persen']):
                    similarity += 0.2
                elif intent['name'] == 'syarat_dokumen' and any(kw in user_input_normalized for kw in ['bayar', 'pembayaran', 'kpr', 'cicil', 'angsuran', 'proses', 'alur', 'tahap', 'langkah', 'sistem']):
                    similarity += 0.3
                elif intent['name'] == 'rekomendasi_proyek' and any(kw in user_input_normalized for kw in ['rekomendasi', 'rekom', 'sarankan', 'saran', 'cocok', 'daerah', 'kawasan']):
                    similarity += 0.2

            similarity = min(similarity, 1.0)
            
            if similarity > highest_score:
                highest_score = similarity
                best_match = intent
                matched_phrase = phrase

    # Adjust threshold for specific intents
    # PERBAIKAN: Turunkan threshold untuk rekomendasi_proyek
    required_score = 0.65 if best_match and best_match['name'] in ['info_proyek', 'rekomendasi_proyek'] else 0.55

    # Generate response
    if best_match and highest_score > required_score:
        print(f"🎯 Best match: {best_match['name']} (score: {highest_score:.2f})")
        response_text = best_match['responses'][0] if best_match['responses'] else ""

        # Detect entities
        entities = detect_entities(user_input)
        project = entities.get('proyek')
        lokasi = entities.get('lokasi')
        print(f"🧩 Detected entities: {entities}")

        # Dapatkan daftar proyek valid
        VALID_PROJECTS = get_valid_projects()

        # ===== FALLBACK 1: Info tanpa proyek =====
        if best_match and best_match['name'] == 'info_proyek':
            # First, handle list requests
            if any(kw in user_input_normalized for kw in ['semua proyek', 'proyek yang ada', 'proyek tersedia', 'apa saja proyek', 'list proyek', 'daftar proyek']):
                projects_list = "\n".join(f"- {p}" for p in VALID_PROJECTS)
                return format_response(
                    "Berikut proyek yang tersedia:\n\n"
                    f"{projects_list}\n\n"
                    "Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
                )
            
            # Then handle cases with no project specified
            if not project:
                # Check if query is specifically asking for project info
                if "info" in user_input_normalized and len(user_input_normalized.split()) <= 2:
                    projects_list = "\n".join(f"- {p}" for p in VALID_PROJECTS)
                    return format_response(
                        "Silakan sebutkan proyek yang ingin Anda ketahui informasinya.\n"
                        "Contoh: 'info Natureland Kiano 3'\n\n"
                        "Proyek yang tersedia:\n"
                        f"{projects_list}"
                    )
                else:
                    # Return the daftar_proyek response for general info requests
                    daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
                    if daftar_intent:
                        return format_response(daftar_intent['responses'][0])
        
        # ===== FALLBACK 2: Proyek tidak valid =====
        if project and not is_valid_project(project):
            # Ambil nama asli dari input user (dari teks normalized)
            user_project = re.search(r'info\s+(.+)', user_input_normalized, re.IGNORECASE)
            project_name = user_project.group(1).strip() if user_project else project
            
            # Handle khusus untuk Kiano 1
            if "kiano 1" in project_name.lower() or "natureland kiano 1" in project_name.lower():
                return format_response(
                    "bold_startNatureland Kiano 1:bold_end\n"
                    "Maaf, proyek ini sudah sold out. Kami merekomendasikan proyek terbaru kami:\n\n"
                    "🏡 Natureland Kiano 2 (Jatisampurna, Bekasi)\n"
                    "🏡 Natureland Kiano 3 (Cibarusah, Bekasi)\n"
                    "🌳 Green Jonggol Village (Jonggol, Bogor)\n\n"
                    "Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
                )
            
            projects_list = "\n".join(f"• {p}" for p in VALID_PROJECTS)
            return format_response(
                "Maaf, proyek yang Anda maksud tidak tersedia di Kianoland Group.\n\n"
                "Proyek yang tersedia:\n"
                f"{projects_list}\n\n"
                "Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
            )
        
        # TAMBAHKAN pengecualian khusus untuk minat_beli:
        if best_match['name'] == 'minat_beli':
            # Gunakan respons daftar proyek, bukan respons asli minat_beli
            daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
            if daftar_intent:
                return format_response(daftar_intent['responses'][0])

        # Handle info_harga intent specifically
        if best_match['name'] == 'info_harga':
            # Jika proyek Kiano 1 terdeteksi
            if project == "Natureland Kiano 1":
                response_text = (
                    "bold_startHarga Natureland Kiano 1:bold_end\n"
                    "Maaf, proyek ini sudah sold out. Berikut harga proyek terbaru kami:\n\n"
                    "🏡 Natureland Kiano 2: Rp 600 Juta\n"
                    "🏡 Natureland Kiano 3: Rp 465 Juta\n"
                    "🌳 Green Jonggol Village: Rp 400 Juta\n\n"
                    "Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
                )
            else:
                # Proses template khusus info_harga
                response_text = process_conditional_templates(response_text, project, lokasi)
            return format_response(response_text)

        # Special handling for location intent
        if best_match['name'] == 'info_lokasi':
            # If no project detected, show project selection message
            if not project:
                return format_response(
                    "Silakan sebutkan proyek yang ingin Anda ketahui lokasinya.\n"
                    "Contoh: 'lokasi Natureland Kiano 3'\n\n"
                    "Proyek yang tersedia:\n"
                    "- Natureland Kiano 2\n"
                    "- Natureland Kiano 3\n"
                    "- Green Jonggol Village"
                )
            
            # Process templates with the detected project
            response_text = process_conditional_templates(response_text, project, lokasi)
            return format_response(response_text)

        # Handle DP/promo questions
        if any(kw in user_input_normalized for kw in ['dp', 'uang muka']):
            project_name = project or "Kianoland"
            response_text = (
                f"Untuk pembelian rumah di {project_name}, kami menyediakan promo khusus:\n\n"
                "• DP 0% untuk KPR BTN Zero\n"
                "• Tanpa Biaya Administrasi\n"
                "• Gratis Kanopi dan Carport\n\n"
                "Info detail promo ketik 'promo'"
            )
        else:
            # PERBAIKAN: Tangani khusus untuk Kiano 1
            if project == "Natureland Kiano 1":
                response_text = (
                    "bold_startNatureland Kiano 1:bold_end\n"
                    "Maaf, proyek ini sudah sold out. Kami merekomendasikan proyek terbaru kami:\n\n"
                    "🏡 Natureland Kiano 2 (Jatisampurna, Bekasi)\n"
                    "🏡 Natureland Kiano 3 (Cibarusah, Bekasi)\n"
                    "🌳 Green Jonggol Village (Jonggol, Bogor)\n\n"
                    "Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
                )

            # PERBAIKAN KHUSUS UNTUK REKOMENDASI_PROYEK
            if best_match['name'] == 'rekomendasi_proyek':

                    # Gunakan lokasi sebagai parameter utama
                    response_text = process_conditional_templates(response_text, lokasi, lokasi)
                    return format_response(response_text)
            else:
                response_text = process_conditional_templates(response_text, project, lokasi)
            
            # Fix duplicate text
            response_text = re.sub(
                r'(?P<project>[\w\s]+) - Detail Proyek:[\s\n]*(bold_start\s*)?\1 - Detail Proyek:',
                r'\1 - Detail Proyek:',
                response_text,
                flags=re.IGNORECASE
            )
        
        print(f"✅ Processed response: {response_text}")
        return format_response(response_text)

    # Deteksi khusus minat beli/kunjungan proyek
    minat_keywords = ['beli', 'kunjung', 'proyek', 'rumah', 'minat']
    if any(kw in user_input_normalized for kw in minat_keywords):
        # Cek apakah ada proyek yang disebutkan
        project = None
        # Use more robust regex patterns that account for word boundaries
        if re.search(r'\b(kiano\s*2|natureland\s*2|nlk2)\b', user_input_normalized):
            project = "Natureland Kiano 2"
        elif re.search(r'\b(kiano\s*3|natureland\s*3|nlk3)\b', user_input_normalized):
            project = "Natureland Kiano 3"
        elif re.search(r'\b(green\s*jonggol|jonggol\s*village|gjv|jonggol)\b', user_input_normalized):
            project = "Green Jonggol Village"
        
        if project:
            # ✅ Validasi proyek
            if not is_valid_project(project):
                projects_list = "\n".join(f"• {p}" for p in get_valid_projects())
                return format_response(
                    f"Maaf, proyek yang Anda maksud tidak tersedia di Kianoland Group.\n\n"
                    f"Proyek yang tersedia:\n{projects_list}\n\n"
                    f"Ketik 'info [nama_proyek]' untuk detail lebih lanjut."
                )

            # Jika valid
            contact = marketing_contacts.get(project, "Bp. Toni - 0896 3823 0725")
            return format_response(
                f"Terima kasih atas minat Anda pada proyek {project}!\n\n"
                f"Untuk proses pembelian atau kunjungan proyek, silakan hubungi marketing kami:\n"
                f"📞 {contact}\n\n"
                "Atau kunjungi website resmi kami:\n"
                "🌐 https://kianolandgroup.com"
            )
        else:
            # Gunakan respons dari daftar_proyek.json
            daftar_intent = next((i for i in INTENTS if i['name'] == 'daftar_proyek'), None)
            if daftar_intent:
                return format_response(daftar_intent['responses'][0])

    # Special fallbacks - only trigger if keywords are prominent
    payment_keywords = ['bayar', 'pembayaran', 'kpr', 'cicil', 'angsuran', 'proses', 'alur', 'tahap', 'langkah']
    if (any(kw in user_input_normalized for kw in payment_keywords) and
        any(kw in user_input_normalized.split() for kw in payment_keywords)):
        payment_intent = next((i for i in INTENTS if i['name'] == 'syarat_dokumen'), None)
        if payment_intent:
            return format_response(payment_intent['responses'][0])
    
    # Replace the current bantuan fallback block with:
    if len(user_input_normalized.split()) <= 4:
        bantuan_keywords = ['bantu', 'tolong', 'panduan', 'help', 'bantuan']
        if any(kw in user_input_normalized for kw in bantuan_keywords):
            bantuan_intent = next((i for i in INTENTS if i['name'] == 'bantuan'), None)
            if bantuan_intent:
                return format_response(bantuan_intent['responses'][0])

    # ===== Logika Fallback yang Diperbaiki =====

    # Jika ada intent yang cocok (walaupun skornya rendah), prioritaskan fallback dari intent tersebut.
    if best_match:
        # Untuk intent spesifik yang butuh nama proyek, minta klarifikasi.
        specific_intents_needing_project = ['info_fasilitas', 'info_harga', 'info_lokasi', 'info_proyek']
        if best_match['name'] in specific_intents_needing_project:
            # Ambil respons dari intent yang terdeteksi
            response_text = best_match['responses'][0] if best_match['responses'] else ""
            # Gunakan template fallback dari intent tersebut, yang akan meminta pengguna menyebutkan proyek
            processed_response = process_conditional_templates(response_text, project=None)
            return format_response(processed_response)

    # Jika setelah semua pengecekan masih tidak ada intent yang jelas, baru gunakan fallback umum.
    fallback_intent = next((i for i in INTENTS if i['name'] == 'default_fallback'), None)
    if fallback_intent:
        return format_response(fallback_intent['responses'][0])
    
    # Fallback terakhir jika file default_fallback.json tidak ada
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