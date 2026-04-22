from __future__ import annotations

import streamlit as st

from src.api import call_uc3m_api
from src.prompts import get_main_rag_prompt, get_summary_prompt, get_suggestion_prompt
from src.system import (
    format_context_for_prompt, 
    load_faiss_bundle, 
    retrieve_context, 
    get_bot_response,           
    detect_language,            
    translate_to_english,      
    translate_response_to_target 
)

st.set_page_config(page_title="UC3M NLP Project", layout="wide")

INDEX_PATH = "data/faiss/dailymed.index"
META_PATH = "data/faiss/dailymed_chunks.jsonl"


@st.cache_resource
def load_rag_system():
    return load_faiss_bundle(INDEX_PATH, META_PATH)
def set_next_query(query):
    """Función callback para que el botón funcione correctamente"""
    st.session_state.next_query = query


model, index, chunks = load_rag_system()

st.title("DailyMed RAG Chatbot")
st.caption("Educational project. Informational only, not medical advice.")

st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=8, value=5)
score_threshold = st.sidebar.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
enable_summary = st.sidebar.toggle("Auto-summary", value=False)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "next_query" not in st.session_state:
    st.session_state.next_query = None

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['drug_name']}** | {src['section_title']} | set_id: `{src['set_id']}` | score: {src['score']:.3f}"
                    )
                    st.write(src["content"])
        
        if msg.get("suggestion") and "Error" not in msg["suggestion"]:
            st.write("💡 **Recommended follow-up:**")
            st.button(
                msg["suggestion"],
                key=f"hist_{i}", 
                on_click=set_next_query,
                args=(msg["suggestion"],)
            )

if st.session_state.next_query:
    user_query = st.session_state.next_query
    st.session_state.next_query = None  # Limpiar el estado
else:
    user_query = st.chat_input("Ask a question about the DailyMed documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    
    from src.system import get_bot_response, detect_language, translate_response_to_target

    # language detection
    original_lang = detect_language(user_query)
    answer, retrieved = get_bot_response(
        user_query, model, index, chunks, top_k, score_threshold
    )

    
    suggestion = None
    summary = None
    
    if retrieved:
        raw_suggestion = call_uc3m_api(get_suggestion_prompt(answer, user_query))
        suggestion = translate_response_to_target(raw_suggestion, original_lang)
        # summary
        if enable_summary:
            raw_summary = call_uc3m_api(get_summary_prompt(answer))
            summary = translate_response_to_target(raw_summary, original_lang)

    assistant_payload = {
        "role": "assistant", 
        "content": answer, 
        "sources": retrieved,
        "suggestion": suggestion,
        "summary": summary
    }

   
    with st.chat_message("assistant"):
        st.markdown(assistant_payload["content"])
        
        # Mostrar botones de sugerencia
        if assistant_payload.get("suggestion") and retrieved:
            st.write("💡 **Recommended follow-up:**")
            st.button(
                assistant_payload["suggestion"],
                key=f"new_res_{len(st.session_state.messages)}",
                on_click=set_next_query,
                args=(assistant_payload["suggestion"],)
            )

        # Mostrar fuentes (Sources)
        if assistant_payload["sources"]:
            with st.expander("Sources"):
                for src in assistant_payload["sources"]:
                    st.markdown(f"**{src['drug_name']}** | {src['section_title']}")
                    st.write(src["content"])

        # Mostrar resumen automático (Extra Feature)
        if assistant_payload.get("summary"):
            st.divider()
            st.info(f"**Summary:** {assistant_payload['summary']}")

    st.session_state.messages.append(assistant_payload)
    st.rerun()
