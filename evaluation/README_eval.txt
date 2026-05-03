# RAG System Evaluation — DailyMed NLP Project (UC3M 2026)

## Overview

This module evaluates the retrieval quality of the DailyMed RAG system across a grid of hyperparameter combinations. It measures how well the FAISS vector index retrieves the right clinical documents before the LLM generates an answer, and produces a CSV + JSON report for every combination tested.

The evaluation is **retrieval-only** — it does not call the LLM. This isolates retrieval errors from generation errors: even a perfect LLM cannot produce a good answer if the retrieval step fails.

---

## Evaluation Criteria

| Criterion | What it measures | How it is computed |
|---|---|---|
| **RAG System Quality** | Whether the system retrieves relevant documents and ranks them well | Hit@K, Recall@K, Precision@K, MRR |
| **Document Coverage** | Fraction of queries answered using database information | Hit@K across all answerable questions |
| **Response Time** | Speed at which retrieval completes | Average and P95 latency in milliseconds |

---

## Metrics

### Retrieval Metrics

All metrics are computed by comparing the **retrieved documents** (drug name + section title) against the **expected support** defined in the evaluation set.

Matching uses fuzzy string comparison (`difflib.SequenceMatcher`, threshold 0.75) plus a section title mapping dictionary that normalises FDA section names (e.g. `"warnings and precautions"`) to internal group labels (e.g. `"safety_risk"`). This makes the evaluation robust to minor naming differences between the eval set and the index.

#### Hit@K
Binary: did at least one retrieved document match any expected document?

```
Hit@K = 1 if |relevant ∩ top-K| ≥ 1, else 0
```

Aggregated as the fraction of questions where at least one relevant document was found.

#### Recall@K
Fraction of relevant documents found in the top-K results. Answers: *did we retrieve everything we needed?*

```
Recall@K = |relevant ∩ top-K| / |relevant|
```

#### Precision@K
Fraction of the top-K retrieved documents that are actually relevant. Answers: *how much noise are we sending to the LLM?*

```
Precision@K = |relevant ∩ top-K| / K
```

#### MRR (Mean Reciprocal Rank)
Rewards systems that rank the best result first. Answers: *is the best result at the top?*

```
MRR = (1/|Q|) × Σ (1 / rank_i)
```

Where `rank_i` is the position (1-indexed) of the first relevant document for query `i`.

### Latency Metrics

| Metric | Description |
|---|---|
| `lat_avg_ms` | Average retrieval time across all queries |
| `lat_p95_ms` | 95th percentile latency — the worst-case speed for 95% of queries |

---

## Evaluation Grid

The script runs every combination of:

| Parameter | Values tested |
|---|---|
| **K** (top-k retrieved) | 3, 5, 8, 10 |
| **Threshold** (min cosine similarity) | 0.20, 0.30, 0.40, 0.60, 0.80 |

This produces **20 combinations** in total. For each combination the script reports metrics for:
- All questions
- English-only questions
- Non-English questions (translated to English before retrieval, matching the production behaviour of `system.py`)

---

## Similarity Score

Retrieval uses **cosine similarity** derived from the L2 distance returned by FAISS. With normalised embeddings the relationship is exact:

```
cosine_similarity = 1.0 - (L2_distance / 2.0)
```

A threshold of 0.25 means only documents with at least 25% cosine similarity to the query are considered. Documents below the threshold are discarded before any metric is computed.

---

## Output Files

After running the evaluation, the following files are produced under `--output-dir` (default: `evaluation/grid_results/`):

```
grid_results/
├── grid_retrieval_comparison.csv       ← Main results table (one row per K×threshold combination)
├── error_analysis_report.txt           ← Missed retrievals at K=10, threshold=0.20
└── combinations/
    ├── k3_t0.20_details.csv            ← Per-question breakdown
    ├── k3_t0.20_summary.json           ← Summary metrics for this combination
    ├── k5_t0.30_details.csv
    ├── k5_t0.30_summary.json
    └── ...                             ← One pair of files per combination
```

### `grid_retrieval_comparison.csv` columns

| Column | Description |
|---|---|
| `k` | Number of documents retrieved |
| `threshold` | Minimum cosine similarity cutoff |
| `hit@k` | Hit rate across all questions |
| `recall@k` | Average Recall@K |
| `precision@k` | Average Precision@K |
| `mrr` | Mean Reciprocal Rank |
| `lat_avg_ms` | Average retrieval latency (ms) |
| `lat_p95_ms` | P95 retrieval latency (ms) |
| `hit@k_en` | Hit rate — English questions only |
| `rec@k_en` | Recall — English questions only |
| `mrr_en` | MRR — English questions only |
| `hit@k_others` | Hit rate — non-English questions |
| `rec@k_others` | Recall — non-English questions |
| `mrr_others` | MRR — non-English questions |

### Per-combination detail CSV columns

| Column | Description |
|---|---|
| `id` | Question ID from eval set |
| `language` | Language of the original question |
| `original_question` | Question as written in the eval set |
| `english_translation` | Translation used for retrieval |
| `expected_support` | List of (drug, section) pairs the system should find |
| `retrieved_documents` | List of (drug, section) pairs actually retrieved |
| `hit` | 1 if at least one match, 0 otherwise |
| `recall` | Recall@K for this question |
| `precision` | Precision@K for this question |
| `mrr` | MRR for this question |
| `latency_ms` | Retrieval time in milliseconds |

---

## Evaluation Set Format

The eval set is a JSONL file where each line is a JSON object:

```json
{
  "id": "q001",
  "question": "What are the contraindications of Aspirin?",
  "language": "en",
  "expected_refusal": false,
  "expected_support": [
    {"drug_name": "Aspirin", "section_title": "contraindications"},
    {"drug_name": "Aspirin", "section_title": "warnings and precautions"}
  ]
}
```

| Field | Description |
|---|---|
| `id` | Unique identifier |
| `question` | The query (any language) |
| `language` | ISO language code (`"en"`, `"es"`, `"fr"`, etc.) |
| `expected_refusal` | `true` if the correct answer is "I don't have information" |
| `expected_support` | List of documents the retriever should find (only for answerable questions) |

Questions with `expected_refusal: true` are excluded from retrieval metrics — they are used only for generation-level refusal accuracy, which is evaluated separately.

---

## How to Run the Evaluation

### 1. Run the Evaluation Grid

```bash
python -m scripts.evaluate_rag \
  --eval-set evaluation/dailymed_eval_v2.jsonl \
  --index data/vector_db/smart_index/index.faiss \
  --meta data/vector_db/smart_index_inspect.jsonl \
  --output-dir evaluation/grid_results
```

### 2. Launch the Dashboard

After running the evaluation, launch the Streamlit dashboard to view tables and line charts interactively:

```bash
streamlit run evaluation/dashboard_unified.py
```

The dashboard shows:
- A filterable comparison table (filter by K and threshold)
- Line charts for Recall@K, Precision@K, MRR, and Average Latency
- A download button to export the filtered results as CSV

---

## Interpreting Results

- **High Recall@K, low Precision@K** → the system retrieves relevant documents but also lots of noise. Consider raising the threshold or lowering K.
- **High MRR** → the best document tends to appear near the top of the results, which is good for the LLM (it sees the most relevant context first).
- **Hit@K drops sharply as threshold increases** → many relevant documents have moderate similarity scores. The production threshold should stay below the point where Hit@K degrades significantly.
- **Foreign language gap** (`hit@k_en` vs `hit@k_others`) → if non-English questions perform worse, the translation step in `system.py` may be introducing errors.