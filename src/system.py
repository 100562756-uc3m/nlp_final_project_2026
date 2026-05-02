from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import faiss
import difflib
import numpy as np
from sentence_transformers import SentenceTransformer
from src.api import call_uc3m_api

# Updated to the model used for the 600k chunks
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_faiss_bundle(
    index_dir: str | Path,
    model_name: str = EMBED_MODEL_NAME,
    ):
    """
    Loads the FAISS index and the corresponding metadata chunks.
    Paths:
    FAISS files are in: data/vector_db/smart_index/
    JSONL file is in:  data/vector_db/
    """
    print(f"Loading Embedding Model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Path to the FAISS binary files
    index_path = os.path.join(index_dir, "index.faiss")
    
    # Path to the JSONL file (which is one level up from the index_dir)
    # index_dir is 'data/vector_db/smart_index', so we go up to 'data/vector_db/'
    parent_dir = os.path.dirname(index_dir)
    meta_path = os.path.join(parent_dir, "smart_index_inspect.jsonl")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing FAISS index at: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing Metadata file at: {meta_path}")

    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    print(f"Loading Metadata from {meta_path} (this may take a minute)...")
    chunks = load_jsonl(meta_path)
    
    return model, index, chunks

def retrieve_context(query, model, index, chunks, k=5, score_threshold=0.30):
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    
    results = []
    seen_content = set() # To prevent duplicates

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks): continue
        if float(score) < score_threshold: continue
        
        content = chunks[idx]["content"]
        # Only add if we haven't seen this EXACT text in this search
        if content not in seen_content:
            item = dict(chunks[idx])
            item["score"] = float(score)
            results.append(item)
            seen_content.add(content)
            
    return results

def retrieve_context(query, model, index, chunks, k=8, score_threshold=0.25):
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k)
    
    results = []
    unique_texts = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or float(score) < score_threshold:
            continue
        
        current_text = chunks[idx]["content"]
        
        # Check if this text is too similar to what we already have
        is_duplicate = False
        for seen_text in unique_texts:
            # If the texts are more than 90% similar, ignore the new one
            ratio = difflib.SequenceMatcher(None, current_text[:500], seen_text[:500]).ratio()
            if ratio > 0.90:
                is_duplicate = True
                break
        
        if not is_duplicate:
            item = dict(chunks[idx])
            item["score"] = float(score)
            results.append(item)
            unique_texts.append(current_text)
            
    return results

def retrieve_context(query, model, index, chunks, k=8, score_threshold=0.25):
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    
    # FAISS devuelve la Distancia L2
    distances, indices = index.search(query_vec, k)
    
    results = []
    unique_texts = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0: continue
        
        # 1. Transformamos la Distancia L2 en un "Similarity Score" (de 0 a 1)
        # Si la distancia es 0, la similitud es 1 (100%).
        # Si la distancia es grande (ej. 3), la similitud es baja (0.25 o 25%).
        similarity = 1.0 / (1.0 + float(dist))
        #similarity coseno
        similarity = 1.0 - (float(dist) / 2.0)
        
        # 2. Ahora SÍ filtramos por similitud: "Si es MENOR que el umbral, lo descarto"
        if similarity < score_threshold:
            continue
        
        current_text = chunks[idx]["content"]
        
        is_duplicate = False
        for seen_text in unique_texts:
            ratio = difflib.SequenceMatcher(None, current_text[:500], seen_text[:500]).ratio()
            if ratio > 0.90:
                is_duplicate = True
                break
        
        if not is_duplicate:
            item = dict(chunks[idx])
            # 3. Guardamos la SIMILITUD en lugar de la distancia
            item["score"] = similarity 
            results.append(item)
            unique_texts.append(current_text)
            
    return results

def assess_retrieval_quality(retrieved: list[dict], weak_threshold: float) -> str:
    """
    Devuelve 'none', 'weak', o 'strong' según los scores de los chunks recuperados.
    - 'none':   no hay chunks
    - 'weak':   hay chunks pero todos están por debajo de weak_threshold
    - 'strong': al menos un chunk supera weak_threshold
    """
    if not retrieved:
        return "none"
    
    best_score = max(item.get("score", 0.0) for item in retrieved)
    
    if best_score < weak_threshold:
        return "weak"
    return "strong"


def format_context_for_prompt(retrieved_chunks: list[dict]) -> str:
    """Formats the metadata and text for the LLM input."""
    lines: list[str] = []
    for i, item in enumerate(retrieved_chunks, start=1):
        # Using the NEW metadata keys: 'drug', 'group', 'part'
        meta = item.get('metadata', {})
        drug = meta.get('drug', 'Unknown Drug')
        group = meta.get('group', 'General Info')
        part = meta.get('part', 1)
        
        lines.append(
            f"--- SOURCE {i} ---\n"
            f"Drug Context: {drug}\n"
            f"Clinical Category: {group} (Part {part})\n"
            f"Content: {item['content']}"
        )
    return "\n\n".join(lines)

# --- Multilanguage Logic ---

def detect_language(query: str) -> str:
    prompt = f"Identify the language of the following text. Respond ONLY with the language name (e.g., 'English', 'Spanish', 'French'): '{query}'"
    lang = call_uc3m_api(prompt)
    return lang.strip().replace(".", "").split()[0] # Take first word to be safe

def translate_to_english(query: str) -> str:
    prompt = f"Translate the following medical query to English. Respond ONLY with the translation: '{query}'"
    translated = call_uc3m_api(prompt)
    return translated.strip().replace('"', '')

def translate_response_to_target(text: str, target_lang: str) -> str:
    if target_lang.lower() == "english":
        return text
    prompt = f"Translate the following medical information to {target_lang}. Maintain medical accuracy. Respond ONLY with the translation: '{text}'"
    return call_uc3m_api(prompt).strip()



def get_bot_response(user_query: str, model, index, chunks, top_k=5, threshold=0.30):
    from src.prompts import get_main_rag_prompt
    
    # 1. Language Handling
    original_lang = detect_language(user_query)
    search_query = translate_to_english(user_query) if original_lang.lower() != "english" else user_query
    
    # 2. Vector Retrieval
    retrieved_chunks = retrieve_context(search_query, model, index, chunks, k=top_k, score_threshold=threshold)
    
    #without info only respond sorry...
    if not retrieved_chunks:
        error_msg = "I'm sorry, I don't have enough information in the document database to answer that."
        return translate_response_to_target(error_msg, original_lang), [], "none"
    # 3. Evaluar calidad ANTES de construir el prompt
    quality = assess_retrieval_quality(retrieved_chunks, weak_threshold=threshold+0.15)  # ← NUEVO

    # 4. LLM Generation
    context_str = format_context_for_prompt(retrieved_chunks)
    final_prompt = get_main_rag_prompt(context_str, search_query, retrieval_quality=quality)
    answer_en = call_uc3m_api(final_prompt)

    #Add that dont show sources if not sufficient data
    #if "I'm sorry, I don't have enough information in the document database to answer that." in answer_en or "don't have enough information" in answer_en:
    #    retrieved_chunks = []
     #   quality = "none"
    REFUSAL_EXACT = "don't have enough information in the document database to answer that."
    if REFUSAL_EXACT in answer_en.lower():
        retrieved_chunks = []
        quality = "none"

    
    # 4. Final Translation
    final_answer = translate_response_to_target(answer_en, original_lang)
    
    return final_answer, retrieved_chunks, quality

def get_language_code(lang_name: str) -> str:
    """Maps language names from detect_language to gTTS ISO codes."""
    mapping = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Hindi": "hi",
        "Thai": "th"
    }
    return mapping.get(lang_name, "en")
