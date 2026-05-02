from __future__ import annotations

import streamlit as st
import time
import os
from gtts import gTTS
import io
from src.system import get_language_code

from src.api import call_uc3m_api
from src.constants import DEFAULT_K, DEFAULT_THRESHOLD
from src.prompts import get_main_rag_prompt, get_summary_prompt, get_suggestion_prompt
from src.system import (
    load_faiss_bundle, 
    get_bot_response,           
    detect_language,             
    translate_response_to_target,
    assess_retrieval_quality 
)

st.set_page_config(page_title="UC3M NLP Project - DailyMed", layout="wide")

# Directory setup
INDEX_DIR = "data/vector_db/smart_index"

def render_audio_button(text, language_name):
    """Generates audio in memory and plays it in Streamlit."""
    try:
        lang_code = get_language_code(language_name)
        tts = gTTS(text=text, lang=lang_code)
        
        # In-memory generation to avoid temporary files
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        st.audio(audio_fp, format='audio/mp3')
    except Exception as e:
        st.error(f"Audio error: {e}")
        
@st.cache_resource
def load_rag_system():
    return load_faiss_bundle(INDEX_DIR)

def set_next_query(query):
    st.session_state.next_query = query

# Initialize system
with st.spinner("Loading clinical database... Please wait."):
    model, reranker, index, chunks = load_rag_system()

# --- HEADER SECTION ---
st.title("DailyMed RAG Chatbot")

# Enhanced Description and Instructions
with st.expander("About this Assistant & How to use", expanded=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **What is this?**  
        This is an advanced Retrieval-Augmented Generation (RAG) system based on the **DailyMed** clinical database. 
        It provides evidence-based information regarding drug indications, dosages, and safety risks.
        
        **Supported Languages:**  
          English, Spanish (Español), French (Français), German (Deutsch), Italian (Italiano), Portuguese (Português), Hindi (हिन्दी), Thai (ไทย).
        
        **⚠️ Disclaimer:**  
        *Educational project only. Information is provided for informational purposes and does not constitute medical advice, diagnosis, or treatment. Always seek the advice of a physician.*
        """)
    with col2:
        st.markdown("**Example Questions:**")
        # Buttons to trigger examples
        if st.button("What is Ibuprofen used for?"):
            st.session_state.next_query = "What is Ibuprofen used for?"
        if st.button("Dosage for Naproxen?"):
            st.session_state.next_query = "What is the recommended dosage for Naproxen?"
        if st.button("Acyclovir safety warnings?"):
            st.session_state.next_query = "What are the safety warnings for Acyclovir?"

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_query" not in st.session_state:
    st.session_state.next_query = None

# --- SIDEBAR ---
st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider("Top-K (Context Chunks)", min_value=1, max_value=10, value=DEFAULT_K)
score_threshold = st.sidebar.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01)
enable_summary = st.sidebar.toggle("Auto-summary", value=False)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.next_query = None
    st.rerun()

# --- EXPORT CHAT IN THE SIDEBAR ---
with st.sidebar:
    st.markdown("### Export")
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_history = ""
        for msg in st.session_state.messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            chat_history += f"=== {role_label.upper()} ===\n"
            chat_history += f"{msg['content']}\n\n"
            if msg["role"] == "assistant" and msg.get("sources"):
                chat_history += "Consulted Sources:\n"
                for idx, s in enumerate(msg["sources"], start=1):
                    meta = s.get("metadata", {})
                    raw_drug = meta.get('drug', '')
                    if not raw_drug or str(raw_drug).strip() in ["", "None", "****"]:
                        drug_name = "FDA Clinical Label"
                    else:
                        drug_name = str(raw_drug)[:60]
                        
                    group = meta.get('group', 'General Info')
                    score = s.get('score', 0.0)
                    content = s.get('content', 'No content available.')
                    
                    # Formato detallado para el .txt
                    chat_history += f"--- SOURCE {idx} ---\n"
                    chat_history += f"Drug: {drug_name} | Group: {group}\n"
                    chat_history += f"Similarity Score: {score:.3f}\n"
                    chat_history += f"Content:\n{content}\n\n"
                
            chat_history += "-" * 40 + "\n\n"
            

        st.download_button(
            label="📥 Download history(.txt)",
            data=chat_history,
            file_name="search_history.txt",
            mime="text/plain"
        )
    else:
        st.info("The history is empty. Make a question first.")


# --- CHAT HISTORY RENDERER ---
# This loop handles both old and new messages for consistent UI
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # ----------Botón de Voz (Audio)------------
        if msg["role"] == "assistant" and msg.get("content"):
            # Solo mostramos el botón si la respuesta no es el mensaje de error/refusal
            if "I'm sorry" not in msg["content"]:
                if st.button("🔊 Listen to response", key=f"voice_{i}"):
                    # Usamos el idioma guardado en el mensaje o inglés por defecto
                    lang_to_use = msg.get("detected_lang", "English")
                    render_audio_button(msg["content"], lang_to_use)
        
        # 1. Timing (Under the text)
        if msg["role"] == "assistant" and msg.get("elapsed"):
            st.caption(f"Response time: {msg['elapsed']:.2f} seconds")

        # Badge de calidad del retrieval  ← NUEVO
        if msg["role"] == "assistant" and msg.get("retrieval_quality"):
            quality = msg["retrieval_quality"]
            if quality == "weak":
                st.warning("Low similarity sources", icon="⚠️")
            

        # 2. Summary (if available)
        if msg.get("summary"):
            st.info(f"**Summary:** {msg['summary']}")

        # 3. Source Display (Cleaned up titles)
        if msg.get("sources"):
            with st.expander("Clinical Sources Consulted"):
                for idx, src in enumerate(msg["sources"], start=1):
                    meta = src.get("metadata", {})

                    # --- FALLBACK LOGIC FOR MISSING DRUG NAMES ---
                    raw_drug = meta.get('drug', '')
                    # Check if the name is missing, empty, or just whitespace
                    if not raw_drug or str(raw_drug).strip() in ["", "None", "****"]:
                        drug_name = "FDA Clinical Label"
                    else:
                        drug_name = str(raw_drug)[:60]
                    
                    score = src.get('score', 0.0)
                    group = meta.get('group', 'General Info')
                    
                    st.markdown(f"**SOURCE {idx}** | {drug_name} | `{group}`")
                    st.caption(f"Similarity Score: `{score:.3f}`")
                    st.info(src["content"])
                    
                    if idx < len(msg["sources"]):
                        st.divider()

        # 4. Two-Button Suggestion Logic
        if msg.get("suggestion") and msg["role"] == "assistant":
            st.write("---")
            st.write("💡 **Recommended follow-up:**")
            
            # This is the key: split the string into a list of questions
            suggestions = msg["suggestion"].split('|')
            
            # Create a column for each suggestion (usually 2)
            cols = st.columns(len(suggestions))
            
            for s_idx, sugg in enumerate(suggestions):
                clean_sugg = sugg.strip()
                if clean_sugg: # Ensure the string isn't empty
                    with cols[s_idx]:
                        st.button(
                            clean_sugg, 
                            key=f"sugg_{i}_{s_idx}", # Unique key for each button
                            on_click=set_next_query, 
                            args=(clean_sugg,),
                            use_container_width=True
                        )

# --- INPUT HANDLING ---
user_input = st.chat_input("Ask about drug interactions, side effects, or dosages...")

# Handle button clicks or manual input
if st.session_state.next_query:
    user_query = st.session_state.next_query
    st.session_state.next_query = None 
    query_triggered = True
elif user_input:
    user_query = user_input
    query_triggered = True
else:
    user_query = None
    query_triggered = False

if query_triggered:
    start_time = time.time()
    
    # 1. Process User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.rerun() # Rerun to show the user message immediately

# Logic to generate assistant response
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    start_time = time.time()
    
    with st.spinner("Analyzing medical documents..."):
        # Detect language early to pass to suggestion prompt
        original_lang = detect_language(last_user_msg)
        
        # Get RAG answer
        answer, retrieved, quality = get_bot_response(
            last_user_msg, model, reranker, index, chunks, top_k, score_threshold
        )

        suggestion = None
        summary = None
        # Inside app.py
        if retrieved:
            try:
                # 2. Get Suggestions 
                found_groups = list(set([src.get('metadata', {}).get('group') for src in retrieved]))
                raw_suggestion = call_uc3m_api(get_suggestion_prompt(answer, user_query, original_lang, found_groups))
                suggestion = raw_suggestion.strip().split('\n')[-1]

                # 3. Summary Logic (Condition: words > 30)
                if enable_summary:
                    raw_summary = call_uc3m_api(get_summary_prompt(answer))
                    summary = translate_response_to_target(raw_summary, original_lang)
            except Exception as e:
                st.error(f"Post-processing error: {e}")
                
        elapsed_time = time.time() - start_time 

        # 4. Create Payload and Append
        assistant_payload = {
            "role": "assistant", 
            "content": answer, 
            "sources": retrieved,
            "suggestion": suggestion,
            "summary": summary,
            "elapsed": elapsed_time,
            "retrieval_quality": quality,
            "detected_lang": original_lang
        }
        st.session_state.messages.append(assistant_payload)
        st.rerun()