"""
This script performs a "Grid Search" evaluation to find the optimal settings for 
the RAG (Retrieval-Augmented Generation) system. It tests different combinations 
of:
  1. Top-K (How many documents to retrieve: 3, 5, 8, 10).
  2. Similarity Threshold (Minimum cosine similarity score: 0.20 to 0.80).

The script simulates the production environment by translating non-English 
queries into English before searching the clinical database (DailyMed).

KEY OUTPUTS:
  - grid_retrieval_comparison.csv: A global matrix of all tested configurations.
  - combinations/: A directory containing detailed CSV logs for every single query 
    under every tested parameter set.
  - error_analysis_report.txt: A breakdown of the hardest questions to answer 
    (failures at the most permissive settings).

HOW TO RUN:
Ensure your PYTHONPATH is set to the project root and run:
    python scripts/evaluate_rag.py --eval-set evaluation/dailymed_eval_v2.jsonl

METRICS CALCULATED:
  - Hit@K: Did the system find at least one correct source?
  - Recall@K: What fraction of all required sources were found?
  - Precision@K: What fraction of retrieved sources were actually relevant?
  - MRR (Mean Reciprocal Rank): How high up in the list was the best source?
  - Latency: Average and 95th percentile (p95) response times in milliseconds.
================================================================================
"""

import os
# Environment optimization for FAISS and Tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json, re, sys, time, csv, difflib
from pathlib import Path

# --- SYSTEM PATH CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
from sentence_transformers import SentenceTransformer

# Custom imports from our project source
from src.system import translate_to_english
from src.constants import SUPER_GROUP_MAP


# --- EVALUATION PARAMETERS ---
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
K_VALUES    = [3, 5, 8, 10]
T_VALUES    = [0.20, 0.30, 0.40, 0.60, 0.80]

# --- TEXT NORMALIZATION HELPERS ---

def normalize(t: str) -> str:
    """
    Standardizes text for comparison.
    Converts to lowercase, removes punctuation, and collapses multiple spaces.
    """
    if not t: return ""
    t = re.sub(r"[^\w\s]", "", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

# Words that appear in drug names but carry no semantic value for matching.
# Removing them reduces false negatives when comparing retrieved vs expected drug names.
DRUG_NOISE_WORDS = [
    "tablets", "tablet", "capsules", "capsule", "injection", "solution",
    "cream", "ointment", "gel", "patch", "suspension", "syrup", "drops",
    "usp", "rx only", "extended release", "oral", "topical", "intravenous",
    "intramuscular", "subcutaneous", "mg", "mcg", "ml", "percent"
]

def normalize_drug_name(name: str) -> str:
    """
    Apply normalize() and additionally strip pharmaceutical noise words
    (dosage forms, units, routes) so that 'Acyclovir Tablets USP 400mg'
    and 'Acyclovir' are treated as the same drug.
    """
    name = normalize(name)
    for w in DRUG_NOISE_WORDS:
        name = re.sub(rf'\b{w}\b', '', name)
    return re.sub(r"\s+", " ", name).strip()

def expected_keys(expected_support: list) -> list:
    """
    Convert the expected_support list from a JSONL eval item into a list of
    (normalized_drug_name, normalized_section) tuples used for matching.
    """
    return [
        (normalize_drug_name(s.get("drug_name", "")), normalize(s.get("section_title", "")))
        for s in expected_support
    ]

def fuzzy_match(str1: str, str2: str, threshold=0.75) -> bool:
    """
    Return True if the two strings are at least `threshold` similar
    according to SequenceMatcher (Levenshtein-like ratio).
    Used to tolerate minor spelling differences in drug names and section titles.
    """
    if not str1 or not str2: return False
    return difflib.SequenceMatcher(None, str1, str2).ratio() >= threshold

def is_match(retrieved_tuple: tuple, expected_tuple: tuple) -> bool:
    """
    Core logic to decide if a retrieved chunk is "Correct".
    Checks both the drug name (partial or fuzzy) and the section super-group.
    """
    r_drug, r_sec = retrieved_tuple
    e_drug, e_sec = expected_tuple  # e_sec is the canonical super-group label from the eval set
    
    # 1. Match the Drug: Substring match or fuzzy match
    drug_match = (e_drug in r_drug) or (r_drug in e_drug) or fuzzy_match(e_drug, r_drug, 0.65)
    
    # 2. Match the Super-Group: Exact or substring match
    sec_match = (e_sec == r_sec) or (e_sec in r_sec) or (r_sec in e_sec)
    
    return drug_match and sec_match

# --- METADATA AND INDEXING ---

def build_idx_map(meta_path: str) -> dict:
    """
    Streams the inspection JSONL and builds a light in-memory map of 
    row_index -> (drug, section).
    This allows us to validate hits without loading the full text of chunks.
    """
    print(f"Reading metadata from {meta_path} (streaming, light)...")
    idx_map = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            
            chunk = json.loads(line)
            meta = chunk.get("metadata", {})

            # Support both old schema (top-level drug_name/section_title)
            # and new schema (nested metadata.drug / metadata.group)
            raw_drug = meta.get("drug", chunk.get("drug_name", ""))
            raw_sec = meta.get("group", meta.get("category", chunk.get("section_title", "")))

            # Convert the raw section string to the SUPER_GROUP_MAP key format
            # Look up the canonical super-group; fall back to the cleaned string if not found
            clean_raw_sec = raw_sec.strip().upper().replace(" ", "_")
            super_group = SUPER_GROUP_MAP.get(clean_raw_sec, clean_raw_sec)
            
            # Store only the two strings we need — not the full chunk content
            idx_map[i] = (normalize_drug_name(raw_drug), normalize(super_group))
            
    print(f"  Done — {len(idx_map)} chunks indexed.")
    return idx_map

def is_toc_chunk(drug: str, sec: str) -> bool:
    """
    Return True if this chunk looks like a table-of-contents entry.
    TOC chunks have no real drug association and pollute retrieval results,
    so they are filtered out during retrieval.
    """
    toc_section_names = {
        "general_info", "table of contents", "contents", 
        "full prescribing information contents",
        "product information", "index"
    }
    if sec in toc_section_names and (not drug or len(drug) < 3):
        return True
    return False

# --- RETRIEVAL ENGINE ---

def retrieve(query, model, index, idx_map, k, threshold):
    """
    Encode the query, search the FAISS index for the top-K nearest neighbors,
    filter by cosine similarity threshold, and return a list of
    (normalized_drug, normalized_section) tuples.
    """
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(vec, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0: continue
        # L2 to Cosine Similarity conversion
        sim = 1.0 - (float(dist) / 2.0)
        if sim < threshold: continue
        
        drug, sec = idx_map.get(int(idx), ("", ""))
        if is_toc_chunk(drug, sec):
            continue
            
        results.append((drug, sec))
        
    return results

# --- METRICS CALCULATIONS ---

def hit_at_k(retrieved_tuples, exp_list): 
    """
    Hit@K: returns True if at least one retrieved chunk matches any expected chunk.
    Binary metric — does not account for how many expected chunks were found.
    """     
    return any(is_match(r, e) for r in retrieved_tuples for e in exp_list)

def recall_at_k(retrieved_tuples, exp_list):
    """
    Recall@K = |relevant ∩ top-K| / |relevant|
    Fraction of the expected chunks that appear in the retrieved top-K results.
    More informative than Hit@K when a question has multiple expected chunks.
    """   
    if not exp_list: return 0.0
    matches = sum(1 for e in exp_list if any(is_match(r, e) for r in retrieved_tuples))
    return matches / len(exp_list)

def precision_at_k(retrieved_tuples, exp_list, k):
    """
    Precision@K = |relevant ∩ top-K| / K
    Fraction of the K retrieved chunks that are actually relevant.
    Penalizes retrieving many irrelevant chunks alongside the relevant ones.
    """
    if not retrieved_tuples: return 0.0
    top_k = retrieved_tuples[:k]
    matches = sum(1 for r in top_k if any(is_match(r, e) for e in exp_list))
    return matches / k

def mrr_score(retrieved_tuples, exp_list):
    """
    MRR (Mean Reciprocal Rank) = 1 / rank of the first relevant result.
    Rewards systems that place the best result at the top of the list.
    MRR = 1.0 if the relevant chunk is rank 1, 0.5 if rank 2, etc.
    Returns 0.0 if no relevant chunk is found in the top-K results.
    """
    for rank, r in enumerate(retrieved_tuples, 1):
        if any(is_match(r, e) for e in exp_list): 
            return 1.0 / rank
    return 0.0

def p95(vals):
    """
    95th percentile of a list of values.
    Used for latency reporting: p95 latency indicates the worst-case
    response time experienced by 95% of queries.
    """
    if not vals: return 0.0
    s = sorted(vals)
    return s[max(0, int(round(0.95 * (len(s) - 1))))]

# --- EVALUATION ENGINE ---

def eval_subset(subset, model, index, idx_map, k, threshold, collect_errors=False):
    """
    Evaluates a specific subset of questions (e.g., all English or all Foreign).
    Collects metrics and optionally logs detailed errors for debugging.
    """
    hits, recs, precs, mrrs, lats = [], [], [], [], []
    failed_logs, detailed_logs = [], []
    
    for q in subset:
        exp = expected_keys(q.get("expected_support", []))
        
        t0  = time.perf_counter()
        # Use English translation if available (simulates multilingual pipeline)
        query_to_search = q.get("question_english", q["question"])
        retrieved_tuples = retrieve(query_to_search, model, index, idx_map, k, threshold)
        lat = (time.perf_counter() - t0) * 1000
        lats.append(lat)

        is_hit = hit_at_k(retrieved_tuples, exp)
        rec = recall_at_k(retrieved_tuples, exp)
        prec = precision_at_k(retrieved_tuples, exp, k)
        mrr = mrr_score(retrieved_tuples, exp)
        
        hits.append(is_hit)
        recs.append(rec)
        precs.append(prec)
        mrrs.append(mrr)
        
        detailed_logs.append({
            "id": q.get("id", "unknown"),
            "language": q.get("language", "en"),
            "original_question": q["question"],
            "english_translation": query_to_search,
            "expected_support": str(exp),
            "retrieved_documents": str(retrieved_tuples),
            "hit": int(is_hit),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "mrr": round(mrr, 4),
            "latency_ms": round(lat, 2)
        })
        
        if collect_errors and not is_hit:
            failed_logs.append({
                "question": q["question"],
                "expected": exp,
                "retrieved": retrieved_tuples
            })
        
    n = len(subset)
    if n == 0: 
        return {"hit": 0.0, "rec": 0.0, "prec": 0.0, "mrr": 0.0, "lat_avg": 0.0, "lat_p95": 0.0}, [], []
    
    metrics = {
        "hit":     round(sum(hits)/n, 4),
        "rec":     round(sum(recs)/n, 4),
        "prec":    round(sum(precs)/n, 4),
        "mrr":     round(sum(mrrs)/n, 4),
        "lat_avg": round(sum(lats)/n, 2),
        "lat_p95": round(p95(lats), 2),
    }
    return metrics, failed_logs, detailed_logs

# --- MAIN EXECUTION ---

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set",  default="evaluation/dailymed_eval_v2.jsonl")
    parser.add_argument("--index",     default="data/vector_db/smart_index/index.faiss")
    parser.add_argument("--meta",      default="data/vector_db/smart_index_inspect.jsonl")
    parser.add_argument("--output-dir",default="evaluation/grid_results")
    args = parser.parse_args()

    # Load and Filter Evaluation Questions
    print("Loading and processing evaluation questions...")
    questions = []
    with open(args.eval_set, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: questions.append(json.loads(line))
    
    # Only answerable questions have expected_support and meaningful retrieval targets
    answerable = [q for q in questions if not q.get("expected_refusal")]
    
    # Simulation: Translate all foreign questions to English via LLM
    print(f"Loaded {len(answerable)} answerable eval questions.")
    print("Translating foreign questions (simulating system.py behavior)...")
    for i, q in enumerate(answerable):
        lang = q.get("language", "en")
        if lang != "en":
            print(f"  Translating question {i+1} from {lang} to English...")
            q["question_english"] = translate_to_english(q["question"])
        else:
            q["question_english"] = q["question"]

    # Subdivide for language-specific reporting
    english_q = [q for q in answerable if q.get("language", "en") == "en"]
    foreign_q = [q for q in answerable if q.get("language", "en") != "en"]

    # Load RAG Components
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading FAISS index from {args.index}...")
    index = faiss.read_index(args.index)

    idx_map = build_idx_map(args.meta)

    # Prepare Output Directories
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combinations_dir = out / "combinations"
    combinations_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_iters = len(K_VALUES) * len(T_VALUES)
    n_iter = 0
    best_errors = []

    # --- GRID SEARCH START --
    print("\nStarting Evaluation Grid...")
    for k in K_VALUES:
        for t in T_VALUES:
            n_iter += 1
            print(f"[{n_iter}/{total_iters}] Evaluating K={k}, Threshold={t:.2f}...")
            
            # Evaluate the three subsets
            r_all, errors, details = eval_subset(answerable, model, index, idx_map, k, t, collect_errors=True)
            r_english, _, _  = eval_subset(english_q,  model, index, idx_map, k, t)
            r_foreign, _, _  = eval_subset(foreign_q,  model, index, idx_map, k, t)
            
            # Save error log for the most permissive config (largest k, lowest threshold)
            # This config has the best theoretical recall, so failures here are genuine hard cases
            if k == 10 and t == 0.20: 
                best_errors = errors
            
            results.append({"k": k, "t": t, "all": r_all, "english": r_english, "foreign": r_foreign})
            
            # Save per-combination detail CSV and summary JSON for dashboard drill-down and error analysis
            file_prefix = f"k{k}_t{t:.2f}"
            detail_csv_path = combinations_dir / f"{file_prefix}_details.csv"
            if details:
                keys = details[0].keys()
                with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
                    dict_writer = csv.DictWriter(f, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(details)
            
            # Save Summary JSON for dashboard usage
            summary_data = {
                "parameters": {"k": k, "threshold": t},
                "metrics_total": r_all,
                "metrics_english": r_english,
                "metrics_foreign": r_foreign
            }
            with open(combinations_dir / f"{file_prefix}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)

    # --- REPORTING ---
    # 1. Hard Error Report
    # Error analysis report (k=10, t=0.20)
    # Useful for understanding which questions are genuinely hard to retrieve,
    # independent of the threshold — these may need better chunking or query rewriting
    error_path = out / "error_analysis_report.txt"
    with open(error_path, "w", encoding="utf-8") as f:
        f.write("=== ERROR ANALYSIS REPORT (K=10, Th=0.20) ===\n")
        for err in best_errors:
            f.write(f"❓ QUESTION: {err['question']}\n")
            f.write(f"🎯 EXPECTED: {err['expected']}\n")
            f.write(f"❌ RETRIEVED:\n")
            for rank, doc in enumerate(err['retrieved'], 1):
                f.write(f"   [{rank}] {doc}\n")
            f.write("-" * 60 + "\n")
    
    # 2. Main Grid Comparison CSV
    csv_path = out / "grid_retrieval_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k","threshold",
                    "hit@k","recall@k","precision@k","mrr","lat_avg_ms","lat_p95_ms",
                    "hit@k_en","rec@k_en","mrr_en",
                    "hit@k_others","rec@k_others","mrr_others"])
        for r in results:
            a, e, fr = r["all"], r["english"], r["foreign"]
            w.writerow([r["k"], r["t"],
                        a["hit"], a["rec"], a["prec"], a["mrr"], a["lat_avg"], a["lat_p95"],
                        e["hit"], e["rec"], e["mrr"],
                        fr["hit"], fr["rec"], fr["mrr"]])

    print("\n" + "="*50)
    print("✅ EVALUATION SUCCESSFULLY COMPLETED")
    print("="*50)
    
    print(f"\n{'K':>4} | {'Thresh':>7} | {'Hit(TOTAL)':>10} | {'Hit(ENG)':>12} | {'Hit(OTHER)':>11}")
    print("-" * 65)
    for r in results:
        a, e, f = r["all"], r["english"], r["foreign"]
        print(f"{r['k']:>4} | {r['t']:>7.2f} | {a['hit']:>10.4f} | {e['hit']:>12.4f} | {f['hit']:>11.4f}")

    # 3. Best Config Ranker (Composite Score)
    print("\n" + "="*50)
    print("TOP 3 BEST CONFIGURATIONS")
    print("="*50)

    # We look for the minimum and maximum latency of the entire grid in order to normalize it
    latencies = [r["all"]["lat_avg"] for r in results]
    lat_min = min(latencies)
    lat_max = max(latencies)

    # We calculate the "Composite Score" as shown in the Dashboard:
    # Recall (35%), MRR (30%), Precision (20%), Latency (15% inverted)
    scored = []
    for r in results:
        a = r["all"]
        
        # Inverse latency normalization (faster = closer to 1.0)
        if lat_max > lat_min:
            lat_norm = 1.0 - (a["lat_avg"] - lat_min) / (lat_max - lat_min)
        else:
            lat_norm = 1.0
            
        score = (0.35 * a["rec"]) + (0.30 * a["mrr"]) + (0.20 * a["prec"]) + (0.15 * lat_norm)
        scored.append((score, r))

    # We order from highest to lowest score
    scored.sort(key=lambda x: x[0], reverse=True)

    medals = ["1º", "2º", "3º"]
    for i, (score, r) in enumerate(scored[:3]):
        a, e, fr = r["all"], r["english"], r["foreign"]
        print(f"\n{medals[i]} Rank {i+1} — K={r['k']}, Threshold={r['t']:.2f}  (score={score:.4f})")
        print(f"   Hit@K={a['hit']}  Recall@K={a['rec']}  Precision@K={a['prec']}  MRR={a['mrr']}")
        print(f"   Latency avg={a['lat_avg']}ms  p95={a['lat_p95']}ms")
        print(f"   English → Hit={e['hit']}  MRR={e['mrr']}")
        print(f"   Foreign → Hit={fr['hit']}  MRR={fr['mrr']}")

    print(f"\n→ Recommended config: K={scored[0][1]['k']}, Threshold={scored[0][1]['t']:.2f}")
    print("="*50)

if __name__ == "__main__":
    main()