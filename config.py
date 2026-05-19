import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "benchmark.db"
REPORTS_DIR = BASE_DIR / "reports"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None

# Qdrant collection names
COLLECTION_SECTIONS = "sections_coarse"
COLLECTION_CHUNKS = "chunks"

# Embedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100

# Chunking
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
SECTION_EMBED_MAX_TOKENS = 256  # tokens from section used for coarse embedding

# LLM via OpenRouter
LLM_MODEL = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0.1
MAX_CONTEXT_TOKENS = 12000  # hard cap for full-context mode

# OpenRouter pricing for gpt-4o-mini (USD per 1M tokens)
INPUT_COST_PER_1M_USD = 0.15
OUTPUT_COST_PER_1M_USD = 0.60

# Retrieval
TOP_K_STANDARD = 5
TOP_K_COARSE_SECTIONS = 3
TOP_K_FINE_CHUNKS = 5

# Ensure runtime dirs exist
for _d in (DATA_DIR, PROCESSED_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
