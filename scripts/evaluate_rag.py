from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import call_uc3m_api
from src.prompts import get_main_rag_prompt
from src.system import format_context_for_prompt, load_faiss_bundle, retrieve_context

DEFAULT_REFUSAL = "I'm sorry, I don't have enough information in the document database to answer that."
REFUSAL_PATTERNS = [
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "insufficient information",
]


def load_eval_questions(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def response_is_refusal(answer: str) -> bool:
    answer_norm = normalize(answer)
    return any(pattern in answer_norm for pattern in REFUSAL_PATTERNS)


def retrieval_hit(retrieved: list[dict], expected_sections: list[str]) -> bool:
    if not expected_sections:
        return False
    normalized_expected = [normalize(x) for x in expected_sections]
    for item in retrieved:
        section = normalize(item.get("section_title", ""))
        if section in normalized_expected:
            return True
    return False


def citation_present(answer: str) -> bool:
    return bool(re.search(r"\[Source\s+\d+\]", answer))


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(round(0.95 * (len(values) - 1)))))
    return values[idx]


def evaluate(
    eval_path: str,
    index_path: str,
    meta_path: str,
    top_k: int,
    score_threshold: float,
    model_name: str,
    output_dir: str,
) -> None:
    questions = load_eval_questions(eval_path)
    embed_model, index, chunks = load_faiss_bundle(index_path, meta_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / "evaluation_results.csv"
    summary_json = output_dir / "evaluation_summary.json"

    rows: list[dict] = []
    total_latencies_ms: list[float] = []
    retrieval_latencies_ms: list[float] = []

    for item in questions:
        qid = item["id"]
        question = item["question"]
        qtype = item["question_type"]
        expected_sections = item.get("expected_sections", [])
        expected_refusal = bool(item.get("expected_refusal", False))

        t0 = time.perf_counter()
        r0 = time.perf_counter()
        retrieved = retrieve_context(
            question,
            model=embed_model,
            index=index,
            chunks=chunks,
            k=top_k,
            score_threshold=score_threshold,
        )
        retrieval_ms = (time.perf_counter() - r0) * 1000

        if not retrieved:
            answer = DEFAULT_REFUSAL
        else:
            context = format_context_for_prompt(retrieved)
            prompt = get_main_rag_prompt(context, question)
            answer = call_uc3m_api(prompt, model_name=model_name)

        total_ms = (time.perf_counter() - t0) * 1000

        retrieval_ok = retrieval_hit(retrieved, expected_sections) if not expected_refusal else None
        refusal_ok = response_is_refusal(answer) if expected_refusal else None

        row = {
            "id": qid,
            "question_type": qtype,
            "language": item.get("language", ""),
            "question": question,
            "expected_refusal": expected_refusal,
            "expected_sections": " | ".join(expected_sections),
            "retrieved_section_titles": " | ".join(x.get("section_title", "") for x in retrieved),
            "retrieved_drug_names": " | ".join(x.get("drug_name", "") for x in retrieved),
            "retrieval_hit_at_k": retrieval_ok,
            "refusal_correct": refusal_ok,
            "citation_present": citation_present(answer),
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "answer": answer,
            "manual_answer_correct": "",
            "manual_grounded": "",
            "manual_citation_correct": "",
        }
        rows.append(row)
        total_latencies_ms.append(total_ms)
        retrieval_latencies_ms.append(retrieval_ms)

    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    answerable_rows = [r for r in rows if not r["expected_refusal"]]
    unanswerable_rows = [r for r in rows if r["expected_refusal"]]

    summary = {
        "num_questions": len(rows),
        "num_answerable": len(answerable_rows),
        "num_unanswerable": len(unanswerable_rows),
        "retrieval_hit_rate_at_k": round(
            sum(bool(r["retrieval_hit_at_k"]) for r in answerable_rows) / max(1, len(answerable_rows)), 4
        ),
        "refusal_accuracy": round(
            sum(bool(r["refusal_correct"]) for r in unanswerable_rows) / max(1, len(unanswerable_rows)), 4
        ),
        "citation_presence_rate_on_answerable": round(
            sum(bool(r["citation_present"]) for r in answerable_rows) / max(1, len(answerable_rows)), 4
        ),
        "avg_total_latency_ms": round(sum(total_latencies_ms) / max(1, len(total_latencies_ms)), 2),
        "median_total_latency_ms": round(statistics.median(total_latencies_ms) if total_latencies_ms else 0.0, 2),
        "p95_total_latency_ms": round(p95(total_latencies_ms), 2),
        "avg_retrieval_latency_ms": round(sum(retrieval_latencies_ms) / max(1, len(retrieval_latencies_ms)), 2),
        "median_retrieval_latency_ms": round(statistics.median(retrieval_latencies_ms) if retrieval_latencies_ms else 0.0, 2),
        "p95_retrieval_latency_ms": round(p95(retrieval_latencies_ms), 2),
        "settings": {
            "top_k": top_k,
            "score_threshold": score_threshold,
            "model_name": model_name,
            "index_path": index_path,
            "meta_path": meta_path,
        },
        "note": "Use the CSV manual_* columns for human grading of answer correctness, groundedness, and citation correctness.",
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved detailed results to:", detailed_csv)
    print("Saved summary to:", summary_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--index", default="data/faiss/dailymed.index")
    parser.add_argument("--meta", default="data/faiss/dailymed_chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--model-name", default="llama3.1:8b")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    evaluate(
        eval_path=args.eval_set,
        index_path=args.index,
        meta_path=args.meta,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
