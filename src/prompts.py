from __future__ import annotations


# def get_main_rag_prompt(context: str, query: str) -> str:
#     return f"""
# You are a document-grounded medical information assistant for a university project.
# CRITICAL RULE: You MUST answer in the same language as the question, do not mix languages. 

# Rules:
# 1. Answer only from the provided sources.
# 2. Do not give medical advice, prescriptions, or recommendations.
# 3. If the sources are missing or do not answer the question, say the following sentence, in the same language as the query, and nothing else:
#    "I'm sorry, I don't have enough information in the document database to answer that."
# 4. If the sources are weak say: "I'm sorry, I don't have enough information to have a well-founded answer but I have found the following:"
# 5. Cite sources using the format [Source X]. Mention the Drug Name and the Clinical Category (e.g., PHARMACOLOGY) if relevant.
# 6. Prefer a concise answer.

# Sources:
# {context}

# Question:
# {query}

# Answer:
# """.strip()

def get_main_rag_prompt(context: str, query: str) -> str:
    return f"""
    You are a professional medical information assistant for a university project, specialized in FDA DailyMed documentation.
    CRITICAL RULE: You MUST answer in the same language as the question. Do not mix languages.

    Rules:
    1. ANSWERING: Use ONLY the provided sources. Synthesize information from ALL relevant sources to provide a complete picture.
    2. CITATIONS: Use inline citations like [Source 1], [Source 1, Source 3]. Mention the Drug Name and Category (e.g., USAGE_CLINICAL) if it helps clarify the context.
    3. NO ADVICE: Do not provide personal medical advice, prescriptions, or clinical recommendations. 
    4. MISSING INFO: If the sources do not contain the answer, respond ONLY with: 
    "I'm sorry, I don't have enough information in the document database to answer that."
    5. WEAK INFO: If the sources contain some relevant information but do not fully answer every aspect of the question, you must start your response with a warning like: "While the provided documents do not cover every detail of your query, here is what they indicate:" (Translate this warning naturally to the language of the query). CRITICAL: Never use the exact phrases "not enough information" or "insufficient information".
    6. CONCISENESS: Be clinical, professional, and concise. Use bullet points for dosage schedules or lists.

    Sources Provided:
    {context}

    Question:
    {query}

    Answer:
    """.strip()


def get_summary_prompt(answer: str) -> str:
    return (
        "Summarize the following answer in one short sentence while preserving the same language:\n\n"
        f"{answer}"
    )


def get_suggestion_prompt(answer: str, original_query: str, language: str, categories: list) -> str:
    categories_str = ", ".join(categories) if categories else "General Information"
    return (
        f"Context:\n"
        f"- Original Query ({language}): '{original_query}'\n"
        f"- Assistant Answer: '{answer}'\n"
        f"- Available Data Categories: {categories_str}\n\n"
        f"Task: Generate exactly TWO follow-up questions in {language}.\n\n"
        "Constraint Checklist:\n"
        "1. Use the SPECIFIC DRUG NAME (No 'it' or 'this').\n"
        "2. Separate exactly with a pipe character (|).\n"
        "3. Output ONLY the questions. No intro, no context.\n"
        "4. Ensure questions are answerable via the 'Available Data Categories'.\n"
        "5. CRITICAL: Do NOT mention 'Sources', 'Source X', or 'the previous answer'. "
        "Questions must be independent and not refer to the current UI state.\n\n"
        "Example Output Format:\n"
        "What is the dosage of Methocarbamol?|What are the side effects of Methocarbamol?\n\n"
        "Final Output:"
    ).strip()