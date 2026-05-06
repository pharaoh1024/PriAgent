from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── LLM ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = "claude-sonnet-4-6"
LLM_TEMPERATURE: float = 0.0  # deterministic for auditing

# ── Embeddings & RAG ───────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # local, no extra API key
FAISS_INDEX_DIR: Path = Path(__file__).parent / "data" / "faiss_index"
RAG_TOP_K: int = 5

# ── Agent hyper-parameters from the paper ──────────────────────────────
TAU_LOCI_MIN: int = 3            # minimum semantic loci to select
TAU_LOCI_MAX: int = 8            # maximum semantic loci
TAU_SIZE_TOKENS: int = 3000      # threshold for code summarization
REFLECTION_CONFIDENCE_THRESHOLD: float = 0.60  # trigger self-reflection below this

# ── Parallelism ─────────────────────────────────────────────────────────
MAX_CONCURRENT_FLOWS: int = 4   # flows processed in parallel by LangGraph Send

# ── Paths ───────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(__file__).parent / "data"
MEMORY_DB_PATH: Path = DATA_DIR / "audit_memory.json"
