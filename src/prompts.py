from __future__ import annotations

def get_main_rag_prompt(context: str, query: str, retrieval_quality: str = "strong") -> str:
    
    quality_instruction = ""
    if retrieval_quality == "weak":
        quality_instruction = """
    RETRIEVAL QUALITY NOTICE: The system has detected that the retrieved sources have LOW similarity scores 
    to the query. This means the information may be only tangentially related.
    You MUST start your response with this warning (translated naturally to the language of the query):
    While the provided documents do not cover every detail of your query, here is what they indicate:" 
    (Translate this warning naturally to the language of the query). CRITICAL: Never use the exact phrases "not enough information" or "insufficient information"."""
    
    return f"""
    You are a professional medical information assistant for a university project, specialized in FDA DailyMed documentation.
    CRITICAL RULE: You MUST answer in the same language as the question. Do not mix languages.
    {quality_instruction}
    Rules:
    1. ANSWERING: Use ONLY the provided sources. Synthesize information from ALL relevant sources to provide a complete picture.
    2. CITATIONS: Use inline citations like [Source 1], [Source 1, Source 3]. Mention the Drug Name and Category if it helps clarify the context.
    3. NO ADVICE: Do not provide personal medical advice, prescriptions, or clinical recommendations.
    4. MISSING INFO: If the sources do not contain the answer, respond ONLY with:
    "I'm sorry, I don't have enough information in the document database to answer that.". DON'T ADD ANYTHING ELSE TO THE ANSWER. 
    
    5. CONCISENESS: Be clinical, professional, and concise. Use bullet points for dosage schedules or lists.

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