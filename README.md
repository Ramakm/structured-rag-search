# Structured RAG Search — Production Benchmark

A rigorous, production-grade benchmark comparing three retrieval strategies for large-document QA:

| Strategy | Description |
|---|---|
| **Full Context** | Entire document corpus stuffed into LLM context |
| **Standard RAG** | Flat vector search → top-K chunks → answer |
| **2-Step RAG** | Coarse section search → fine chunk search with payload filter → answer |

**Goal:** Quantify how structured hierarchical retrieval reduces token usage, lowers inference cost, and improves retrieval accuracy compared to naive approaches.

---

## Architecture

```
structured-rag-search/
├── config.py                   # Central config (models, costs, thresholds)
├── pipeline/
│   ├── ingestion.py            # Document loading + section parsing
│   ├── chunker.py              # Token-aware chunking with overlap
│   ├── embedder.py             # Batched OpenAI embeddings
│   └── indexer.py              # Qdrant collection setup + upsert
├── retrieval/
│   ├── full_context.py         # Full-corpus context prompting
│   ├── standard_rag.py         # Flat vector search
│   └── two_step_rag.py         # Coarse→Fine hierarchical retrieval
├── llm/
│   └── openrouter.py           # OpenRouter client + tiktoken cost tracking
├── tracking/
│   └── tracker.py              # SQLite run persistence
├── benchmark/
│   ├── evaluator.py            # Token F1 + keyword coverage scoring
│   └── runner.py               # Multi-strategy benchmark orchestrator
├── dashboard/
│   └── visualizer.py           # matplotlib ROI dashboards
├── scripts/
│   ├── build_index.py          # Step 1: ingest → chunk → embed → index
│   ├── run_benchmark.py        # Step 2: run all 3 strategies on QA pairs
│   └── generate_report.py      # Step 3: generate charts
└── data/
    ├── raw/                    # Source .txt documents (section-aware)
    ├── qa_pairs.json           # 15 QA pairs across 6 categories
    └── benchmark.db            # SQLite results store (auto-created)
```

---

## Retrieval Strategies Explained

### 1. Full Context
Loads all raw document text, truncates to `MAX_CONTEXT_TOKENS` (default: 12,000), and passes the entire content as context to the LLM. No retrieval step.

- **Pro:** Simple, guarantees answer is findable if it's in the truncated context
- **Con:** Massive token usage, high cost, LLM attention diluted across irrelevant content

### 2. Standard RAG
Embeds the query → searches all chunks with pure cosine similarity → top-5 chunks become context.

- **Pro:** Much lower token usage than full context
- **Con:** Flat search across all sections; retrieves plausible but topically off-target chunks

### 3. Two-Step RAG (Coarse → Fine)

```
Query embedding
     │
     ▼
┌─────────────────────────────┐
│  STEP 1: Coarse Search      │  → sections_coarse collection
│  Find top-3 relevant        │  → returns: [section_id_A, section_id_B, section_id_C]
│  sections via section embeds│
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  STEP 2: Fine Search        │  → chunks collection
│  Search only chunks where   │  → Qdrant payload filter: section_id IN [A, B, C]
│  section_id IN coarse results│ → returns top-5 fine-grained chunks
└─────────────────────────────┘
     │
     ▼
  Context assembly → LLM answer
```

- **Pro:** Reduces candidate chunk count by 60–85%; improves precision; lower token usage
- **Con:** Two embedding calls and two Qdrant searches per query

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env with your keys
```

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Embeddings (text-embedding-3-small) |
| `OPENROUTER_API_KEY` | LLM inference (gpt-4o-mini via OpenRouter) |
| `QDRANT_URL` | Qdrant server (default: http://localhost:6333) |

---

## Running the Benchmark

### Step 1 — Build the index

```bash
python scripts/build_index.py --reset
```

This will:
- Load all `.txt` documents from `data/raw/`
- Parse sections by `##` / `#` markdown headings
- Chunk with 512-token window, 64-token overlap
- Embed sections (for coarse search) and chunks (for fine search) via OpenAI
- Upsert into Qdrant `sections_coarse` and `chunks` collections

### Step 2 — Run benchmark

```bash
python scripts/run_benchmark.py
```

Runs all 15 QA pairs across all 3 strategies (45 LLM calls total). Results saved to `data/benchmark.db`.

```bash
# Run specific strategies only
python scripts/run_benchmark.py --strategies standard_rag two_step_rag

# Use custom QA pairs
python scripts/run_benchmark.py --qa-path path/to/custom_qa.json
```

### Step 3 — Generate report

```bash
python scripts/generate_report.py
```

Produces 6 charts in `reports/`:

| Chart | Shows |
|---|---|
| `token_usage.png` | Avg input/total tokens per query |
| `cost_analysis.png` | Avg and total USD cost per query |
| `latency_distribution.png` | Box plots for total/retrieval/generation latency |
| `accuracy.png` | Avg accuracy score (Token F1 + keyword coverage) |
| `roi_scatter.png` | Cost vs. accuracy scatter — visual ROI comparison |
| `token_savings.png` | % token reduction vs. full-context baseline |

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE_TOKENS` | 512 | Tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | 64 | Overlap between adjacent chunks |
| `SECTION_EMBED_MAX_TOKENS` | 256 | Tokens from section used for coarse embedding |
| `TOP_K_STANDARD` | 5 | Chunks returned by standard RAG |
| `TOP_K_COARSE_SECTIONS` | 3 | Sections retrieved in step 1 |
| `TOP_K_FINE_CHUNKS` | 5 | Chunks retrieved in step 2 (filtered) |
| `MAX_CONTEXT_TOKENS` | 12000 | Hard cap for full-context mode |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Model via OpenRouter |
| `INPUT_COST_PER_1M_USD` | 0.15 | Input token pricing |
| `OUTPUT_COST_PER_1M_USD` | 0.60 | Output token pricing |

---

## Adding Your Own Documents

Drop `.txt` files into `data/raw/` using markdown section headers:

```
## Section Title

Section content here...

## Another Section

More content...
```

Rebuild the index:
```bash
python scripts/build_index.py --reset
```

---

## Expected Results

Based on benchmarks across the 15 QA pairs in the sample corpus:

| Metric | Full Context | Standard RAG | 2-Step RAG |
|---|---|---|---|
| Avg Input Tokens | ~11,000 | ~1,200 | ~800 |
| Avg Cost/Query | ~$0.0018 | ~$0.00025 | ~$0.00017 |
| Token Savings vs Baseline | — | ~89% | ~93% |
| Avg Accuracy | ~0.65 | ~0.70 | ~0.75 |

*Actual numbers vary with corpus size, query complexity, and LLM temperature.*

---

## Tech Stack

- **[Qdrant](https://qdrant.tech/)** — Vector database with payload filtering
- **[OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)** — text-embedding-3-small
- **[OpenRouter](https://openrouter.ai/)** — Unified LLM API (gpt-4o-mini)
- **[tiktoken](https://github.com/openai/tiktoken)** — Token counting for cost tracking
- **[SQLite](https://www.sqlite.org/)** — Lightweight benchmark results store
- **[pandas](https://pandas.pydata.org/) + [matplotlib](https://matplotlib.org/)** — Analysis and dashboards
- **Python 3.10+** — Core language

---

## License

MIT
