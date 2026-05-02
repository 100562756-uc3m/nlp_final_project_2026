import json, time, csv, sys
from pathlib import Path
from src.system import get_bot_response
from src.api import call_uc3m_api
from src.constants import DEFAULT_K, DEFAULT_THRESHOLD 

# Add root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def get_evaluator_prompt(question, context, answer):
    return f"""
    You are a Medical Quality Auditor for a RAG system. 
    Evaluate the following answer based on the provided clinical context.

    ### Question: {question}
    ### Context: {context}
    ### Generated Answer: {answer}

    Provide a score from 1 to 5 for:
    1. GROUNDEDNESS: Is the answer fully supported by the context? (1=Hallucination, 5=Perfectly Grounded)
    2. RELEVANCE: Does it answer the user's question? (1=Irrelevant, 5=Perfectly Relevant)

    Format your response EXACTLY like this:
    Groundedness: [score]
    Relevance: [score]
    Reasoning: [one sentence explanation]
    """

def run_gen_eval(eval_set_path, output_path):
    # Load models/index (Similar to your app.py setup)
    # This assumes you have access to the load_rag_system logic
    from app import load_rag_system
    model, reranker, index, chunks = load_rag_system()

    with open(eval_set_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f]

    results = []
    
    for q in questions:
        print(f"Evaluating {q['id']}...")
        start_t = time.time()
        
        # 1. Run the actual RAG pipeline using global constants
        answer, retrieved, quality = get_bot_response(
            q['question'], model, reranker, index, chunks, 
            top_k=DEFAULT_K, 
            threshold=DEFAULT_THRESHOLD
        )
        latency = time.time() - start_t

        # 2. Check for Refusal
        is_refusal = "I'm sorry, I don't have enough information" in answer
        refusal_ok = (is_refusal == q['expected_refusal'])

        # 3. If it's not a refusal, let's judge the quality
        g_score, r_score, reason = 0, 0, "Refusal Case"
        if not is_refusal and retrieved:
            context_text = "\n".join([c['content'] for c in retrieved])
            eval_resp = call_uc3m_api(get_evaluator_prompt(q['question'], context_text, answer))
            
            # Parse scores using regex
            try:
                g_score = int(re.search(r"Groundedness: (\d)", eval_resp).group(1))
                r_score = int(re.search(r"Relevance: (\d)", eval_resp).group(1))
                reason = re.search(r"Reasoning: (.*)", eval_resp).group(1)
            except:
                reason = "Error parsing evaluator response"

        results.append({
            "id": q['id'],
            "question": q['question'],
            "expected_refusal": q['expected_refusal'],
            "actual_refusal": is_refusal,
            "refusal_correct": refusal_ok,
            "groundedness": g_score,
            "relevance": r_score,
            "latency": round(latency, 2),
            "reasoning": reason
        })

    # Save to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_gen_eval("evaluation/dailymed_eval_v2.jsonl", "evaluation/generation_results.csv")