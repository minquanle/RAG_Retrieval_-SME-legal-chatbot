# R2AI — Vietnamese Legal QA System

A legal question-answering system for Vietnamese small and medium enterprises (SMEs), consisting of two independent notebooks executed sequentially on Kaggle:

| Step | Notebook | Description |
|---|---|---|
| 1 | `pipeline-corpus.ipynb` | Retrieval: find relevant legal articles for 2,000 questions |
| 2 | `llm-gen.ipynb` | LLM generation: use Qwen3-14B to generate answers with citations |

---

## System Architecture

### Step 1 — Retrieval Pipeline (`pipeline-corpus.ipynb`)

```
Corpus (6,151 legal articles)
    ├── BM25 (underthesea + 4-char n-gram, k1=1.5, b=0.4) → top-100 articles
    ├── Dense BGE-M3 (clause-level encoding) → top-200 clauses → MaxRank → top-50 articles
    └── Query Enrichment (keywords + sub-queries) → BM25 extra + Dense extra
                ↓ RRF fusion (k=60) → pool of 50 candidates
    BGE-reranker-v2-m3 Sliding Window MaxP (chunk 1200 chars, overlap 200)
                ↓ Adaptive Soft-Floor cutoff (ABS_FLOOR=0.38, MAX_K=7, MIN_K=1)
    → retrieval_v6.json + submission.zip
```

### Step 2 — LLM Generation (`llm-gen.ipynb`)

```
Old submissions (10 zip files) → Voting ensemble → Top-10 article candidates per question
    + Corpus lookup (SME merged + th1nhng0 fallback)
                ↓
    Qwen3-14B (4-bit NF4 quantization, transformers)
    System prompt: Legal QA with citation rules
    (Optional) Review pass: LLM-as-a-Judge
                ↓ Checkpoint JSONL after each question
    → results_qwen3_14b_llm_filter.json + .zip
```

---

## Environment Requirements

| Component | Requirement |
|---|---|
| Platform | **Kaggle Notebooks** (recommended) |
| Step 1 runtime | GPU T4 x2 or P100 (≥ 16 GB total VRAM) |
| Step 2 runtime | GPU T4 x2 or P100/A100 (≥ 16 GB VRAM with 4-bit quant) |
| Python | 3.10+ (Kaggle default) |
| Internet connection | **Enabled** (required for downloading models from HuggingFace) |

---

## Input Data

The two notebooks use **two different data sources**.

### Step 1 — Retrieval: Kaggle Dataset

The dataset is available on Kaggle at: `letuano5/r2ai-corpus`

The notebook uses exactly the following 5 files (plus 1 optional):

| File | Size | Description |
|---|---|---|
| `corpus_luat_sme_merged_v3.jsonl` | 14.18 MB | Main SME corpus (~5,658 articles) |
| `patch_laws_articles.jsonl` | 820 kB | 5 additional laws 2024–2025 (273 articles) |
| `sme_clauses_v4.jsonl` | 51.57 MB | Clause chunks of the SME corpus (~41,746 chunks) |
| `patch_laws_clauses.jsonl` | 2.89 MB | Clause chunks of the patch laws |
| `R2AIStage1DATA.json` | 531 kB | 2,000 test questions |
| `r2ai_enrichment.jsonl` | 2.39 MB | *(Optional)* Query enrichment cache |

Path in the notebook (Cell 3):
```python
DATASET_DIR = "/kaggle/input/datasets/letuano5/r2ai-corpus"
```

### Step 2 — LLM: Google Drive

Download data from Google Drive:
**https://drive.google.com/drive/folders/1oxWNBI49W7C5RfEVMiyl8SyxpV1JJhvw?usp=drive_link**

Then upload to **two separate Kaggle datasets**:

**Dataset 1: `r2ai-corpus`** (shared with Step 1, add file):

| File | Description |
|---|---|
| `corpus_luat_sme_merged_v3.jsonl` | SME corpus (already present from Step 1) |
| `th1nhng0_articles_v2.jsonl` | Fallback article corpus (~517 MB) |

**Dataset 2: `r2ai-old-submissions`** (create new):

| Content | Description |
|---|---|
| 10 old submission `.zip` files | Each zip contains `results.json` with `relevant_articles` |

Paths in the notebook (config cell):
```python
SUBMISSIONS_DIR = "/kaggle/input/r2ai-old-submissions"
SME_MERGED      = "/kaggle/input/r2ai-corpus/corpus_luat_sme_merged_v3.jsonl"
TH_ARTICLES     = "/kaggle/input/r2ai-corpus/th1nhng0_articles_v2.jsonl"
```

---

## Models (auto-downloaded from HuggingFace)

| Model | Size | Used in |
|---|---|---|
| `BAAI/bge-m3` | ~2.27 GB | Step 1 — Dense embedder |
| `BAAI/bge-reranker-v2-m3` | ~2.27 GB | Step 1 — Reranker |
| `Qwen/Qwen3-14B` | ~9 GB (4-bit) / ~28 GB (fp16) | Step 2 — LLM generation |

To avoid rate-limit warnings when downloading models, add the Kaggle Secret `HF_TOKEN`:
```
Kaggle Notebook → Add-ons → Secrets → New Secret
Key: HF_TOKEN | Value: hf_xxxxxxxxxxxxxxxxxxxx
```

---

## How to Run

### Step 1 — `pipeline-corpus.ipynb`

**Kaggle notebook setup:**
1. Create a new notebook on Kaggle, upload `pipeline-corpus.ipynb`
2. Settings → Accelerator: select `GPU T4 x2` (or P100)
3. Settings → Internet: **On**
4. Add Data → search for the `letuano5/r2ai-corpus` dataset → Add

**Run the cells in order:**

| Cell | Action | Estimated time |
|---|---|---|
| Cell 1 | Install packages: `sentence-transformers`, `rank_bm25`, `underthesea` | ~1 min |
| Cell 2 | Import libraries, check GPU | < 30 sec |
| Cell 3 | Load config, verify files exist (5 ✓ marks) | < 5 sec |
| Cell 4 | Load SME corpus + patch (6,151 articles) + clause chunks (41,746 chunks) | ~1 min |
| Cell 5 | Build BM25 index *(cached — fast from the 2nd run)* | ~5 min first run |
| Cell 6 | Load BGE-M3 + encode 41,746 clauses *(cached)* | ~15 min first run |
| Cell 7 | Encode 2,000 queries | ~2 min |
| Cell 8 | Load enrichment + encode sub-queries (~3,030 texts) | ~5 min |
| Cell 9 | BM25 + Dense retrieval → RRF fusion (2,000 queries) | ~3 min |
| Cell 10 | Load BGE-reranker + rerank 2,000 queries | ~30–60 min |
| Cell 11 | Save `retrieval_v6.json` + create `submission.zip` | < 1 min |

**Output in `/kaggle/working/`:**
- `retrieval_v6.json` — full retrieval results
- `results.json` — submission format (with empty `answer` field)
- `submission.zip` — submission file

> **Cache note:** Cells 5 and 6 automatically save the BM25/embedding cache to `/kaggle/working/cache/`. If you rerun the notebook (without restarting the session), these two cells skip encoding and load from cache — saving ~20 min.

---

### Step 2 — `llm-gen.ipynb`

**Kaggle notebook setup:**
1. Create a new notebook, upload `llm-gen.ipynb`
2. Settings → Accelerator: `GPU T4 x2` or P100/A100
3. Settings → Internet: **On**
4. Add Data → `letuano5/r2ai-corpus` (must contain `corpus_luat_sme_merged_v3.jsonl` + `th1nhng0_articles_v2.jsonl`)
5. Add Data → dataset containing the 10 old-submission zip files (create the `r2ai-old-submissions` dataset)

**Before running — verify 2 important variables in the config cell:**
```python
# Make sure these match your Kaggle dataset names
SUBMISSIONS_DIR = "/kaggle/input/r2ai-old-submissions"   # contains 10 .zip files
TH_ARTICLES     = "/kaggle/input/r2ai-corpus/th1nhng0_articles_v2.jsonl"
```

**Run the cells in order:**

| Cell | Action | Estimated time |
|---|---|---|
| Cell install | *(Optional)* Set `INSTALL_PACKAGES = True` if packages need installing | ~2 min |
| Cell imports | Import libraries, check CUDA | < 30 sec |
| Cell config | Set paths, hyperparameters, print `REPO_ROOT` and file paths | < 5 sec |
| Cell prompts | Define system prompt + review prompt (just assigns variables) | < 5 sec |
| Cell helpers | Define utility functions | < 5 sec |
| Cell load submissions | Load 10 zips, vote articles → `questions` dict | ~1 min |
| Cell corpus lookup | Build full-text lookup table from SME corpus + fallback | ~2 min |
| Cell checkpoint | Load checkpoint if a previous run was interrupted (auto-resume) | < 5 sec |
| Cell prepare | Prepare LLM messages (vote + lookup text) | ~1 min |
| Cell load model | Load Qwen3-14B with 4-bit NF4 quantization | ~10–15 min |
| Cell generate | Generate answers for 2,000 questions (checkpoint after each) | ~3–6 hrs |
| Cell export | Export `results_qwen3_14b_llm_filter.json` + `.zip` | < 1 min |
| Cell validate | Verify the output format end-to-end | < 1 min |

**Output in `/kaggle/working/`:**
- `submissions/results_qwen3_14b_llm_filter.json` — full results
- `submissions/results_qwen3_14b_llm_filter.zip` — submission file
- `data/eval/qwen3_14b_llm_filter_checkpoint.jsonl` — checkpoint (append-only JSONL)

> **Resume:** If the notebook is interrupted mid-run, just run it again from the top — the checkpoint automatically skips already-processed questions. To start over from scratch, set `IGNORE_CHECKPOINT = True` in the config cell.

---

## Configuration and Tuning

### pipeline-corpus.ipynb — Main parameters (Cell 3)

```python
# BM25
BM25_K1 = 1.5     # Raised from default 1.2 — suits legal texts with intentional repetition
BM25_B  = 0.4     # Lowered from default 0.75 — penalises long articles less
BM25_K  = 100     # Top-K articles from BM25

# Dense retrieval (clause-level)
DENSE_CLAUSE_K = 200   # Top-K clauses
DENSE_K        = 50    # Number of articles after grouping clauses

# Reranker cutoff
ABS_FLOOR = 0.38  # Minimum absolute score threshold (lower → higher recall)
MAX_K     = 7     # Return at most 7 articles
MIN_K     = 1     # Return at least 1 article (don't force 2 when only 1 is correct)

# Query enrichment
USE_ENRICHMENT = True   # Use r2ai_enrichment.jsonl
USE_KEYWORDS   = True   # Add a BM25 leg from keywords
USE_DECOMP     = True   # Add BM25 + Dense legs from sub-queries
USE_HYDE       = False  # Don't use HyDE (disabled to avoid noise)
```

### llm-gen.ipynb — Main parameters (config cell)

```python
MODEL_ID       = "Qwen/Qwen3-14B"  # Can switch to Qwen3-7B if VRAM is tight
LOAD_IN_4BIT   = True               # 4-bit NF4 quant (~9 GB VRAM); False = fp16 (~28 GB)
MAX_ARTICLES   = 10                 # Number of candidate articles injected into the LLM prompt
MIN_VOTE       = 1                  # Minimum vote threshold from old submissions
REVIEW_PASS    = False              # True = run LLM-as-a-Judge (doubles runtime)
THINKING       = "off"              # "on" = enable Qwen3 chain-of-thought (slower)
TEMPERATURE    = 0                  # Greedy decoding (0 = deterministic)
RETRIES        = 1                  # Number of retries on LLM errors
ARTICLE_CHARS  = 6000               # Truncate each article's full_text to 6,000 chars
MAX_INPUT_TOKENS = 24576            # Maximum context window
MAX_NEW_TOKENS   = 1600             # Maximum output length
```

---

## Output Format

Each entry in `results.json`:

```json
{
  "id": 1,
  "question": "Legal question...",
  "answer": "Under Article X..., the enterprise must...",
  "relevant_docs": ["doc_id|Document title"],
  "relevant_articles": ["doc_id|Document title|Article X"]
}
```

Format constraints (checked automatically in the validate cell):
- `relevant_articles`: each item must contain exactly **2 `|` characters** (3 parts)
- `relevant_docs`: each item must contain exactly **1 `|` character** (2 parts)

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.1.0 | GPU inference |
| `transformers` | ≥ 4.51.0 | Load / run Qwen3-14B |
| `accelerate` | ≥ 0.33.0 | Multi-GPU device_map |
| `bitsandbytes` | ≥ 0.43.0 | 4-bit NF4 quantization |
| `sentence-transformers` | ≥ 2.7.0 | BGE-M3 embedder + BGE-reranker-v2-m3 |
| `rank-bm25` | ≥ 0.2.2 | BM25Okapi sparse retrieval |
| `underthesea` | ≥ 6.8.0 | Vietnamese word tokenizer |
| `numpy` | ≥ 1.24.0 | Vector operations |
| `tqdm` | ≥ 4.66.0 | Progress bars |
| `json`, `zipfile`, `pathlib`, `hashlib`, `pickle` | stdlib | File I/O, caching |

---

## Common Troubleshooting

**`✗` file not found when running Cell 3 (Step 1)**
→ Check the Kaggle dataset name in `DATASET_DIR`. Open the notebook's "Input" tab to confirm the actual path.

**CUDA out of memory during rerank (Cell 10)**
→ The notebook automatically reduces `batch_sz` on OOM. If it still fails, set `RERANK_BATCH = 16` in Cell 10.

**Step 2 notebook can't find `th1nhng0_articles_v2.jsonl`**
→ Verify the file is uploaded to the Kaggle dataset. The filename must match exactly (including `_v2`).

**Step 2 notebook can't find zip files in `old_submission`**
→ Verify `SUBMISSIONS_DIR` points to the correct Kaggle dataset name that holds the `.zip` files.

**Qwen3-14B loads slowly or errors out**
→ Make sure Internet is enabled on Kaggle. Add `HF_TOKEN` to Kaggle Secrets if you hit a rate limit.

**Want to smoke-test quickly**
→ In the `llm-gen.ipynb` config cell, set `LIMIT = 5` to run only the first 5 questions.
