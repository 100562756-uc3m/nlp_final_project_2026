import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json, re, sys, time, csv, difflib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
from sentence_transformers import SentenceTransformer

from src.system import translate_to_english
# Importamos el mapeo de Super-Grupos para traducir lo que devuelve FAISS
from src.constants import SUPER_GROUP_MAP

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
K_VALUES    = [3, 5, 8, 10]
T_VALUES    = [0.20, 0.30, 0.40, 0.60, 0.80]

# ── Helpers y Limpieza de Fármacos ────────────────────────────────────────────
def normalize(t: str) -> str:
    if not t: return ""
    t = re.sub(r"[^\w\s]", "", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()

# ¡Restauramos tu lista de palabras fantasma para los fármacos!
DRUG_NOISE_WORDS = [
    "tablets", "tablet", "capsules", "capsule", "injection", "solution",
    "cream", "ointment", "gel", "patch", "suspension", "syrup", "drops",
    "usp", "rx only", "extended release", "oral", "topical", "intravenous",
    "intramuscular", "subcutaneous", "mg", "mcg", "ml", "percent"
]

def normalize_drug_name(name: str) -> str:
    name = normalize(name)
    for w in DRUG_NOISE_WORDS:
        name = re.sub(rf'\b{w}\b', '', name)
    return re.sub(r"\s+", " ", name).strip()

def expected_keys(expected_support: list) -> list:
    return [
        (normalize_drug_name(s.get("drug_name", "")), normalize(s.get("section_title", "")))
        for s in expected_support
    ]

def fuzzy_match(str1: str, str2: str, threshold=0.75) -> bool:
    if not str1 or not str2: return False
    return difflib.SequenceMatcher(None, str1, str2).ratio() >= threshold

def is_match(retrieved_tuple: tuple, expected_tuple: tuple) -> bool:
    r_drug, r_sec = retrieved_tuple # r_sec ya viene traducido a Super-Grupo desde idx_map
    e_drug, e_sec = expected_tuple  # e_sec es el Super-Grupo del JSON (ej. "usage_clinical")
    
    # 1. Match de Droga (usando las drogas limpias de USP, etc.)
    drug_match = (e_drug in r_drug) or (r_drug in e_drug) or fuzzy_match(e_drug, r_drug, 0.65)
    
    # 2. Match de Sección directa (Super-Group vs Super-Group)
    sec_match = (e_sec == r_sec) or (e_sec in r_sec) or (r_sec in e_sec)
    
    return drug_match and sec_match

# ── Carga Inteligente del Metadata de FAISS ───────────────────────────────────
def build_idx_map(meta_path: str) -> dict:
    print(f"Reading metadata from {meta_path} (streaming, light)...")
    idx_map = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            
            chunk = json.loads(line)
            meta = chunk.get("metadata", {})
            raw_drug = meta.get("drug", chunk.get("drug_name", ""))
            
            # Cogemos la sección que haya en la BD (puede ser "group", "category" o el título)
            raw_sec = meta.get("group", meta.get("category", chunk.get("section_title", "")))
            
            # TRUCO: Convertimos lo que haya en la BD (ej. "DOSAGE AND ADMINISTRATION") 
            # a formato constante ("DOSAGE_AND_ADMINISTRATION") para buscar su Super-Grupo
            clean_raw_sec = raw_sec.strip().upper().replace(" ", "_")
            super_group = SUPER_GROUP_MAP.get(clean_raw_sec, clean_raw_sec)
            
            # Guardamos el fármaco LIMPIO y la sección traducida a Super-Grupo ("usage_clinical")
            idx_map[i] = (normalize_drug_name(raw_drug), normalize(super_group))
            
    print(f"  Done — {len(idx_map)} chunks indexed.")
    return idx_map

def is_toc_chunk(drug: str, sec: str) -> bool:
    toc_section_names = {
        "general_info", "table of contents", "contents", 
        "full prescribing information contents",
        "product information", "index"
    }
    if sec in toc_section_names and (not drug or len(drug) < 3):
        return True
    return False

# ── Recuperación FAISS ────────────────────────────────────────────────────────
def retrieve(query, model, index, idx_map, k, threshold):
    vec = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(vec, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0: continue
        sim = 1.0 - (float(dist) / 2.0)
        if sim < threshold: continue
        
        drug, sec = idx_map.get(int(idx), ("", ""))
        if is_toc_chunk(drug, sec):
            continue
            
        results.append((drug, sec))
        
    return results

# ── Métricas ──────────────────────────────────────────────────────────────────
def hit_at_k(retrieved_tuples, exp_list):      
    return any(is_match(r, e) for r in retrieved_tuples for e in exp_list)
def recall_at_k(retrieved_tuples, exp_list):   
    if not exp_list: return 0.0
    matches = sum(1 for e in exp_list if any(is_match(r, e) for r in retrieved_tuples))
    return matches / len(exp_list)
def precision_at_k(retrieved_tuples, exp_list, k): 
    if not retrieved_tuples: return 0.0
    top_k = retrieved_tuples[:k]
    matches = sum(1 for r in top_k if any(is_match(r, e) for e in exp_list))
    return matches / k
def mrr_score(retrieved_tuples, exp_list):
    for rank, r in enumerate(retrieved_tuples, 1):
        if any(is_match(r, e) for e in exp_list): 
            return 1.0 / rank
    return 0.0
def p95(vals):
    if not vals: return 0.0
    s = sorted(vals)
    return s[max(0, int(round(0.95 * (len(s) - 1))))]

# ── Runner de Evaluación ──────────────────────────────────────────────────────
def eval_subset(subset, model, index, idx_map, k, threshold, collect_errors=False):
    hits, recs, precs, mrrs, lats = [], [], [], [], []
    failed_logs, detailed_logs = [], []
    
    for q in subset:
        exp = expected_keys(q.get("expected_support", []))
        
        t0  = time.perf_counter()
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

# ── Ejecución Principal ───────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set",  default="evaluation/dailymed_eval_v2.jsonl")
    parser.add_argument("--index",     default="data/vector_db/smart_index/index.faiss")
    parser.add_argument("--meta",      default="data/vector_db/smart_index_inspect.jsonl")
    parser.add_argument("--output-dir",default="evaluation/grid_results")
    args = parser.parse_args()

    print("Loading and processing evaluation questions...")
    questions = []
    with open(args.eval_set, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: questions.append(json.loads(line))
            
    answerable = [q for q in questions if not q.get("expected_refusal")]
    
    print(f"Loaded {len(answerable)} answerable eval questions.")
    print("Translating foreign questions (simulating system.py behavior)...")
    
    for i, q in enumerate(answerable):
        lang = q.get("language", "en")
        if lang != "en":
            print(f"  Translating question {i+1} from {lang} to English...")
            q["question_english"] = translate_to_english(q["question"])
        else:
            q["question_english"] = q["question"]

    english_q = [q for q in answerable if q.get("language", "en") == "en"]
    foreign_q = [q for q in answerable if q.get("language", "en") != "en"]

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading FAISS index from {args.index}...")
    index = faiss.read_index(args.index)

    idx_map = build_idx_map(args.meta)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    combinations_dir = out / "combinations"
    combinations_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_iters = len(K_VALUES) * len(T_VALUES)
    n_iter = 0
    best_errors = []

    print("\nStarting Evaluation Grid...")
    for k in K_VALUES:
        for t in T_VALUES:
            n_iter += 1
            print(f"[{n_iter}/{total_iters}] Evaluating K={k}, Threshold={t:.2f}...")
            
            r_all, errors, details = eval_subset(answerable, model, index, idx_map, k, t, collect_errors=True)
            r_english, _, _  = eval_subset(english_q,  model, index, idx_map, k, t)
            r_foreign, _, _  = eval_subset(foreign_q,  model, index, idx_map, k, t)
            
            if k == 10 and t == 0.20: 
                best_errors = errors
            
            results.append({"k": k, "t": t, "all": r_all, "english": r_english, "foreign": r_foreign})

            file_prefix = f"k{k}_t{t:.2f}"
            
            detail_csv_path = combinations_dir / f"{file_prefix}_details.csv"
            if details:
                keys = details[0].keys()
                with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
                    dict_writer = csv.DictWriter(f, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(details)
            
            summary_data = {
                "parameters": {"k": k, "threshold": t},
                "metrics_total": r_all,
                "metrics_english": r_english,
                "metrics_foreign": r_foreign
            }
            with open(combinations_dir / f"{file_prefix}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4)

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

if __name__ == "__main__":
    main()