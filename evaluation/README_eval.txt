## Evaluation

This project includes a corpus-specific evaluation set for the DailyMed RAG chatbot.

The evaluation is designed to test three things:

1. **Answerable questions**  
   Questions that should be answered using evidence from the DailyMed document database.

2. **Hard / paraphrased questions**  
   Questions that ask for the same information in a less direct way, sometimes in another language, to test retrieval robustness and multilingual behavior.

3. **Unanswerable / refusal questions**  
   Questions that the system should refuse because they require diagnosis, personal medical advice, treatment recommendations, or information not supported by the corpus.

### Final evaluation set

The main evaluation file is based on the actual processed corpus in `data/dailymed_chunks.jsonl`, not on guessed examples.

**Files**
- `evaluation/dailymed_eval_final.jsonl`
- `evaluation/dailymed_eval_final.csv` (optional preview / inspection file)

**Size**
- 60 questions total
- 20 easy answerable
- 20 hard or paraphrased answerable
- 20 unanswerable / refusal

**Languages included**
- English (`en`)
- Spanish (`es`)
- French (`fr`)
- German (`de`)
- Italian (`it`)
- Portuguese (`pt`)
- Hindi (`hi`)
- Thai (`th`)

### Why this evaluation set is better than the starter set

The earlier multilingual starter set was useful for initial testing, but it was not guaranteed to match the real processed DailyMed corpus.  
The final evaluation set is stronger because every answerable question is tied to an exact chunk from our current database.

Each answerable item includes:
- `chunk_id`
- `drug_name`
- `section_title`
- `set_id`
- `document_id`
- `source_path`
- `evidence_preview`

This makes the evaluation more reliable for retrieval and grounding because we know which chunk should support the answer.

### Evaluation file format

Each JSONL row contains metadata such as:

- `id`
- `question_type`
- `difficulty`
- `language`
- `question`
- `expected_refusal`
- `expected_support`
- `expected_drug_name`
- `expected_section_title`

For refusal cases, the file may also include:
- `expected_refusal_reason`

### What is being measured

The evaluation should be used to measure:

#### 1. Retrieval quality
Whether the correct supporting chunk appears in the retrieved results.

Examples:
- Hit@k
- Recall@k
- Whether the expected `chunk_id` or expected section appears in the top-k results

#### 2. Answer grounding
Whether the generated answer is actually supported by the retrieved DailyMed text.

Examples:
- Correct source cited
- Correct drug and section referenced
- No unsupported claims added by the model

#### 3. Refusal behavior
Whether the system refuses questions that should not be answered.

Examples:
- Personal medical advice
- Best drug / best substitute recommendations
- Diagnosis requests
- Personalized dosing requests

#### 4. Multilingual behavior
Whether the chatbot can retrieve and answer correctly when the question is asked in another supported language.

#### 5. Latency
How long the full pipeline takes.

Examples:
- average latency
- median latency
- p95 latency

### How to run the evaluation

Example command:

```bash
python scripts/evaluate_rag.py \
  --eval-set evaluation/dailymed_eval_final.jsonl \
  --top-k 5 \
  --score-threshold 0.30 \
  --output-dir evaluation/results_final_k5_t030