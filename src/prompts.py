from __future__ import annotations


def get_main_rag_prompt(context: str, query: str) -> str:
    return f"""
You are a document-grounded medical information assistant for a university project.
CRITICAL RULE: You MUST answer in the same language as the question, do not mix languages. 
Rules:
1. Answer only from the provided sources.
2. Do not give medical advice, prescriptions, or recommendations.
3. If the sources are missing or do not answer the question, say the following sentence, in the same language as the query,  and nothing else:
   "I'm sorry, I don't have enough information in the document database to answer that."
4. If the sources are weak say: "I'm sorry, I don't have enough information to have a well-founded answer but I have found the following:"
5. Cite the supporting sources inline using the format [Source 1], [Source 2], etc.
6. Prefer a concise answer.

Sources:
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


def get_suggestion_prompt(answer: str, original_query: str) -> str:
    return (
        f"The user original question was: '{original_query}'.\n"
        f"The assistant answer was: '{answer}'.\n\n"
        "Based on this, suggest ONE short follow-up question. "
        "CRITICAL: You MUST write the suggestion in the SAME LANGUAGE as the user's original question. "
        "If the question was in English, the suggestion MUST be in English. "
        "If the question was in Spanish, the suggestion MUST be in Spanish. "
        "Respond ONLY with the question, no extra text."
    )

#5. Answer in the same language as the question.
