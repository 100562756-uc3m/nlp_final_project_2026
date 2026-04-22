from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.api import call_uc3m_api

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_faiss_index(
    chunks: Iterable[dict],
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 128,
):
    chunks = list(chunks)
    model = SentenceTransformer(model_name)
    index = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [x["content"] for x in batch]
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if index is None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    if index is None:
        raise ValueError("No chunks were provided to build the index.")

    return model, index, chunks


def save_faiss_bundle(index, chunks: list[dict], index_path: str | Path, meta_path: str | Path) -> None:
    index_path = Path(index_path)
    meta_path = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with meta_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def load_faiss_bundle(
    index_path: str | Path,
    meta_path: str | Path,
    model_name: str = EMBED_MODEL_NAME,
):
    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(index_path))
    chunks = load_jsonl(meta_path)
    return model, index, chunks


def retrieve_context(
    query: str,
    model,
    index,
    chunks: list[dict],
    k: int = 5,
    score_threshold: float = 0.30,
) -> list[dict]:
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_vec, k)
    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < score_threshold:
            continue
        item = dict(chunks[idx])
        item["score"] = float(score)
        results.append(item)
    return results


def format_context_for_prompt(retrieved_chunks: list[dict]) -> str:
    lines: list[str] = []
    for i, item in enumerate(retrieved_chunks, start=1):
        lines.append(
            f"Source {i}\n"
            f"Drug: {item['drug_name']}\n"
            f"Section: {item['section_title']}\n"
            f"Set ID: {item['set_id']}\n"
            f"Effective Time: {item.get('effective_time', '')}\n"
            f"Content: {item['content']}"
        )
    return "\n\n".join(lines)

#multilanguage
def detect_language(query: str) -> str:
    """Detect language of query"""
    prompt = f"Identify the language of the following text. Respond ONLY with the language name (e.g., 'English', 'Spanish', 'French'): '{query}'"
    lang = call_uc3m_api(prompt)
    return lang.strip().replace(".", "")

def translate_to_english(query: str) -> str:
    """Translate to English"""
    prompt = f"Translate the following medical query to English. Respond ONLY with the translation: '{query}'"
    translated = call_uc3m_api(prompt)
    return translated.strip().replace('"', '')

def translate_response_to_target(text: str, target_lang: str) -> str:
    """Translate answer to original language fo the query"""
    if target_lang.lower() == "english":
        return text
    prompt = f"Translate the following medical information to {target_lang}. Maintain the medical accuracy and tone. Respond ONLY with the translation: '{text}'"
    return call_uc3m_api(prompt).strip()



def get_bot_response(user_query: str, model, index, chunks, top_k=5, threshold=0.30):
    from src.api import call_uc3m_api
    from src.prompts import get_main_rag_prompt
    
    # Language detection
    original_lang = detect_language(user_query)
    
    # Translate 
    if original_lang.lower() != "english":
        search_query = translate_to_english(user_query)
    else:
        search_query = user_query
    
    # retrieve
    retrieved_chunks = retrieve_context(search_query, model, index, chunks, k=top_k, score_threshold=threshold)
    
    if not retrieved_chunks:
        error_msg = "I'm sorry, I don't have enough information in the document database to answer that."
        return translate_response_to_target(error_msg, original_lang), []

    # answer
    context_str = format_context_for_prompt(retrieved_chunks)
    final_prompt = get_main_rag_prompt(context_str, search_query)
    answer_en = call_uc3m_api(final_prompt)
    
    # translate answer
    final_answer = translate_response_to_target(answer_en, original_lang)
    
    return final_answer, retrieved_chunks

