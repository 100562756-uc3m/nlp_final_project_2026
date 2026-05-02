from __future__ import annotations

import streamlit as st
import time
import os

from src.api import call_uc3m_api
from src.prompts import get_main_rag_prompt, get_summary_prompt, get_suggestion_prompt
from src.system import (
    load_faiss_bundle, 
    get_bot_response,           
    detect_language,             
    translate_response_to_target,
    assess_retrieval_quality 
)

st.set_page_config(page_title="UC3M NLP Project", layout="wide")

# Directory setup
INDEX_DIR = "data/vector_db/smart_index"

@st.cache_resource
def load_rag_system():
    return load_faiss_bundle(INDEX_DIR)

def set_next_query(query):
    st.session_state.next_query = query

# Initialize system
with st.spinner("Loading clinical database... Please wait."):
    model, index, chunks = load_rag_system()

st.title("DailyMed RAG Chatbot")
st.caption("Educational project. Informational only, not medical advice.")

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_query" not in st.session_state:
    st.session_state.next_query = None

# --- SIDEBAR ---
st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider("Top-K (Context Chunks)", min_value=1, max_value=10, value=5)
score_threshold = st.sidebar.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
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


# --- INITIALIZE STATE ---
#if "messages" not in st.session_state:
#    st.session_state.messages = []
#if "next_query" not in st.session_state:
#    st.session_state.next_query = None

# --- CHAT HISTORY RENDERER ---
# This loop handles both old and new messages for consistent UI
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
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
            last_user_msg, model, index, chunks, top_k, score_threshold
        )

        suggestion = None
        summary = None
        # Inside app.py
        if retrieved:
            # 2. Get Suggestions 
            found_groups = list(set([src.get('metadata', {}).get('group') for src in retrieved]))
            raw_suggestion = call_uc3m_api(get_suggestion_prompt(answer, user_query, original_lang, found_groups))
            suggestion = raw_suggestion.strip().split('\n')[-1]

            # 3. Summary Logic (Condition: words > 30)
            if enable_summary:
                raw_summary = call_uc3m_api(get_summary_prompt(answer))
                summary = translate_response_to_target(raw_summary, original_lang)
                
        elapsed_time = time.time() - start_time 

        # 4. Create Payload and Append
        assistant_payload = {
            "role": "assistant", 
            "content": answer, 
            "sources": retrieved,
            "suggestion": suggestion,
            "summary": summary,
            "elapsed": elapsed_time,
            "retrieval_quality": quality
        }
        st.session_state.messages.append(assistant_payload)
        st.rerun()

# from __future__ import annotations

# import streamlit as st
# import time
# import os

# from src.api import call_uc3m_api
# from src.prompts import get_main_rag_prompt, get_summary_prompt, get_suggestion_prompt
# from src.system import (
#     load_faiss_bundle, 
#     get_bot_response,           
#     detect_language,             
#     translate_response_to_target 
# )

# st.set_page_config(page_title="UC3M NLP Project", layout="wide")

# # This points to the folder CONTAINING the .faiss file
# INDEX_DIR = "data/vector_db/smart_index"

# @st.cache_resource
# def load_rag_system():
#     # This function in src/system.py must be the 'corrected' version 
#     # that looks for the .jsonl in the parent directory.
#     return load_faiss_bundle(INDEX_DIR)

# def set_next_query(query):
#     st.session_state.next_query = query

# # Initialize system (This will show progress in your terminal)
# with st.spinner("Loading clinical database... Please wait."):
#     model, index, chunks = load_rag_system()

# st.title("DailyMed RAG Chatbot")
# st.caption("Educational project. Informational only, not medical advice.")

# # --- SIDEBAR ---
# st.sidebar.header("Retrieval Settings")
# top_k = st.sidebar.slider("Top-K (Context Chunks)", min_value=1, max_value=10, value=5)
# score_threshold = st.sidebar.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
# enable_summary = st.sidebar.toggle("Auto-summary", value=False)

# if st.sidebar.button("Clear Chat"):
#     st.session_state.messages = []
#     st.rerun()

# # --- INITIALIZE STATE ---
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "next_query" not in st.session_state:
#     st.session_state.next_query = None

# # --- CHAT HISTORY ---
# for i, msg in enumerate(st.session_state.messages):
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
        
#         # 1. Improved Source Display
#         if msg.get("sources"):
#             with st.expander("Clinical Sources Consulted"):
#                 for idx, src in enumerate(msg["sources"], start=1):
#                     meta = src.get("metadata", {})
#                     drug_name = meta.get('drug', 'Unknown')[:50]
#                     score = src.get('score', 0.0)
                    
#                     # Header for each source
#                     st.markdown(f"#### SOURCE {idx}")
#                     st.caption(f"**Drug:** {drug_name} | **Category:** {meta.get('group')} | **Score:** `{score:.3f}`")
                    
#                     # Content inside a container to prevent "Big Headers" from breaking layout
#                     st.info(src["content"])
#                     if idx < len(msg["sources"]):
#                         st.divider()

#         # 2. Display Metadata and Timing
#         if msg["role"] == "assistant" and msg.get("elapsed"):
#             st.caption(f"Response time: {msg['elapsed']:.2f} seconds")

#         # 3. New Two-Button Suggestion Logic
#         if msg.get("suggestion") and msg["role"] == "assistant":
#             # Split the suggestions by the pipe character we defined in the prompt
#             suggestions = msg["suggestion"].split('|')
#             if suggestions:
#                 st.write("**Recommended follow-up:**")
#                 cols = st.columns(len(suggestions))
#                 for s_idx, sugg in enumerate(suggestions):
#                     with cols[s_idx]:
#                         st.button(
#                             sugg.strip(), 
#                             key=f"sugg_{i}_{s_idx}", 
#                             on_click=set_next_query, 
#                             args=(sugg.strip(),),
#                             use_container_width=True # Makes buttons look better
#                         )

# # --- INPUT HANDLING ---
# user_input = st.chat_input("Ask about drug interactions, side effects, or dosages...")

# if st.session_state.next_query:
#     user_query = st.session_state.next_query
#     st.session_state.next_query = None 
# elif user_input:
#     user_query = user_input
# else:
#     user_query = None

# if user_query:
#     start_time = time.time()
#     st.session_state.messages.append({"role": "user", "content": user_query})
#     with st.chat_message("user"):
#         st.markdown(user_query)

#     # 1. Logic via system.py
#     with st.spinner("Analyzing medical documents..."):
#         answer, retrieved = get_bot_response(
#             user_query, model, index, chunks, top_k, score_threshold
#         )

#     # 2. Suggestions & Summaries
#     suggestion = None
#     summary = None
#     original_lang = detect_language(user_query)
    
#     if retrieved:
#         raw_suggestion = call_uc3m_api(get_suggestion_prompt(answer, user_query))
#         suggestion = translate_response_to_target(raw_suggestion, original_lang)
        
#         if enable_summary:
#             raw_summary = call_uc3m_api(get_summary_prompt(answer))
#             summary = translate_response_to_target(raw_summary, original_lang)
            
#     elapsed_time = time.time() - start_time 

#     # 3. Payload
#     assistant_payload = {
#         "role": "assistant", 
#         "content": answer, 
#         "sources": retrieved,
#         "suggestion": suggestion,
#         "summary": summary,
#         "elapsed": elapsed_time
#     }

#     # 4. Display
#     with st.chat_message("assistant"):
#         st.markdown(assistant_payload["content"])
        
#         if assistant_payload.get("suggestion"):
#             st.write("---")
#             st.button(f" {assistant_payload['suggestion']}", key=f"new_{len(st.session_state.messages)}", 
#                       on_click=set_next_query, args=(assistant_payload["suggestion"],))

#         if assistant_payload["sources"]:
#             with st.expander("Clinical Sources Consulted"):
#                 for src in assistant_payload["sources"]:
#                     meta = src.get("metadata", {})
#                     st.markdown(f"**{meta.get('drug')}** | {meta.get('group')} (Part {meta.get('part')})")
#                     st.write(src["content"])

#         if assistant_payload.get("summary"):
#             st.info(f"**Summary:** {assistant_payload['summary']}")

#         st.caption(f" Response time: {assistant_payload['elapsed']:.2f} seconds")

#     st.session_state.messages.append(assistant_payload)