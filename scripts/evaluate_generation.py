import json
import time
import csv
import sys
import re  
from pathlib import Path

# Add root to path so src and app are discoverable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.system import get_bot_response
from src.api import call_uc3m_api
from src.constants import DEFAULT_K, DEFAULT_THRESHOLD 

def get_evaluator_prompt(question, context, answer):
    """Prompt for the LLM-as-a-Judge to grade medical accuracy."""
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
    # Load models/index (imports here to avoid global loading issues)
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

        # 2. Check for Refusal (Consistency check with system message)
        is_refusal = "I'm sorry, I don't have enough information" in answer
        refusal_ok = (is_refusal == q['expected_refusal'])

        # 3. If it's not a refusal, let's judge the quality
        g_score, r_score, reason = 0, 0, "Refusal Case"
        
        if not is_refusal and retrieved:
            context_text = "\n".join([c['content'] for c in retrieved])
            eval_resp = call_uc3m_api(get_evaluator_prompt(q['question'], context_text, answer))
            
            # Flexible Parsing logic to avoid "Error parsing" failures
            try:
                g_match = re.search(r"Groundedness:\s*(\d)", eval_resp, re.IGNORECASE)
                r_match = re.search(r"Relevance:\s*(\d)", eval_resp, re.IGNORECASE)
                reason_match = re.search(r"Reasoning:\s*(.*)", eval_resp, re.IGNORECASE)

                g_score = int(g_match.group(1)) if g_match else 0
                r_score = int(r_match.group(1)) if r_match else 0
                reason = reason_match.group(1).strip() if reason_match else eval_resp.strip().replace("\n", " ")
                
                if g_score == 0 or r_score == 0:
                    reason = f"Parsing Warning: Could not find scores in: {eval_resp[:50]}..."
            except Exception as e:
                reason = f"Parsing Error: {str(e)}"

        # Save results including language for the dashboard
        results.append({
            "id": q['id'],
            "question": q['question'],
            "language": q.get('language', 'en'), # Added for dashboard_gen.py
            "expected_refusal": q['expected_refusal'],
            "actual_refusal": is_refusal,
            "refusal_correct": refusal_ok,
            "groundedness": g_score,
            "relevance": r_score,
            "latency": round(latency, 2),
            "reasoning": reason
        })

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_gen_eval("evaluation/dailymed_eval_v2.jsonl", "evaluation/generation_results.csv")