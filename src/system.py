from __future__ import annotations

import json
import os
import faiss
import difflib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder 
from src.api import call_uc3m_api
from src.constants import DEFAULT_K, DEFAULT_THRESHOLD


# Models
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" 

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

def load_faiss_bundle(index_dir: str | Path):
    """Loads Bi-Encoder, Cross-Encoder, FAISS index, and Metadata."""
    print(f"Loading Bi-Encoder: {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    
    print(f"Loading Cross-Encoder: {RERANK_MODEL_NAME}...")
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    
    # Path to the FAISS binary files
    index_path = os.path.join(index_dir, "index.faiss")
    
    # Path to the JSONL file
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
    
    return model, reranker, index, chunks

# --- Retrieval & Reranking ---
def retrieve_context(query, model, reranker, index, chunks, k=DEFAULT_K, score_threshold=DEFAULT_THRESHOLD):
    """
    Two-stage retrieval: 
    1. Bi-Encoder search (FAISS) for candidates.
    2. Cross-Encoder re-ranking for final selection.
    """
    # STAGE 1: Expanded Search
    initial_k = k * 2 
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_vec, initial_k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0: continue
        
        content = chunks[idx].get("content", "")
        
        # Ignore "noise" chunks that are too short to contain clinical meaning
        if len(content.strip()) < 30:
            continue
            
        # Convert L2 distance to Cosine Similarity
        similarity = 1.0 - (float(dist) / 2.0)
        
        # Filter by similarity 
        if similarity < score_threshold: 
            continue
        
        candidates.append(chunks[idx])

    if not candidates:
        return []
    
    # STAGE 2: Re-ranking
    # Prepare pairs: [query, document_content]
    pairs = [[query, c["content"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    
    # Attach scores and sort by semantic relevance
    for idx, score in enumerate(rerank_scores):
        candidates[idx]["rerank_score"] = float(score)
        # Normalize score for UI display (using sigmoid to put it in 0-1 range)
        candidates[idx]["score"] = 1.0 / (1.0 + np.exp(-score)) 

    # Sort descending by rerank score
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # STAGE 3: Metadata-Aware Deduplication
    final_results = []
    seen_keys = set() 

    for item in candidates:
        meta = item.get("metadata", {})
        drug = meta.get("drug", "Unknown")
        # Optimization: 200 chars is enough to identify a duplicate section
        content_snippet = item["content"][:200] 
        
        is_duplicate = False
        # Only compare against snippets from the SAME drug
        for seen_drug, seen_text in seen_keys:
            if drug == seen_drug:
                # difflib is slow on long strings; 200 chars is much faster
                ratio = difflib.SequenceMatcher(None, content_snippet, seen_text).ratio()
                if ratio > 0.95:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            final_results.append(item)
            seen_keys.add((drug, content_snippet))
            
        if len(final_results) >= k:
            break
            
    return final_results

# --- Quality & Formatting ---

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

# --- Translation & Language Logic ---

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

# --- Main Logic ---
def get_bot_response(user_query: str, model, reranker, index, chunks, top_k=DEFAULT_K, threshold=DEFAULT_THRESHOLD):
    from src.prompts import get_main_rag_prompt
    
    # 1. Handle Language
    original_lang = detect_language(user_query)
    search_query = translate_to_english(user_query) if original_lang.lower() != "english" else user_query
    
    # 2. Retrieve & Rerank
    retrieved_chunks = retrieve_context(search_query, model, reranker, index, chunks, k=top_k, score_threshold=threshold)
    
    #without info only respond sorry...
    if not retrieved_chunks:
        error_msg = "I'm sorry, I don't have enough information in the document database to answer that."
        return translate_response_to_target(error_msg, original_lang), [], "none"
    
    # 3. Assess Quality
    quality = assess_retrieval_quality(retrieved_chunks, weak_threshold=threshold+0.15)  

    # 4. Generate Answer
    context_str = format_context_for_prompt(retrieved_chunks)
    final_prompt = get_main_rag_prompt(context_str, search_query, retrieval_quality=quality)
    answer_en = call_uc3m_api(final_prompt)

    # Add that dont show sources if not sufficient data
    REFUSAL_EXACT = "don't have enough information in the document database to answer that."
    if REFUSAL_EXACT in answer_en.lower():
        retrieved_chunks = []
        quality = "none"

    # 5. Final Translation
    final_answer = translate_response_to_target(answer_en, original_lang)
    
    return final_answer, retrieved_chunks, quality