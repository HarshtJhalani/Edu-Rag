"""All settings in one place. Plain module constants; values chosen in notebooks 02-07."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LIBRARY_DIR = DATA_DIR / "library"                # <role>/<source name>/: original file, source.json, previews/
SOURCE_FILE = "source.json"
INDEX_DIR = DATA_DIR / "index"                    # everything here can be rebuilt from the library
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
CHUNK_SETTINGS_FILE = INDEX_DIR / "settings.json"
CACHE_DIR = INDEX_DIR / "embed_cache"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_FILE = INDEX_DIR / "bm25.pkl"
WEB_CACHE_DIR = DATA_DIR / "cache" / "web"
EVAL_DIR = DATA_DIR / "eval"
APP_SETTINGS_FILE = DATA_DIR / "app_settings.json"

ROLES = ["textbook", "note", "web", "pyq", "syllabus"]
EVIDENCE_ROLES = ["textbook", "note", "web"]      # only these may support an answer

# --- Ollama ---------------------------------------------------------------
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and Ollama listens on IPv4 only, which added
# ~2 s to EVERY request (Ollama reported 0.28 s for an embedding the client waited 2.4 s for).
OLLAMA_URL = "http://127.0.0.1:11434"
LLM = "qwen2.5:7b-instruct"
LLM_SMALL = "qwen2.5:3b-instruct"                 # picked by setup_check when VRAM is under 7 GB
EMBED_MODEL = "bge-m3"                            # best on the 68-question eval (04)
VISION_MODEL = "qwen3-vl:4b-instruct-q4_K_M"
RERANKER = "BAAI/bge-reranker-base"               # CPU; evaluation only (04/05), not used by the app
NUM_CTX = 8192                                    # always set: Ollama silently truncates otherwise
TEMPERATURE = 0.2

QWEN_INSTRUCT = "Instruct: Given a student's question, retrieve textbook passages that answer it\nQuery: "
EMBEDDERS = {
    "qwen3-embedding:0.6b": {"query_prefix": QWEN_INSTRUCT, "doc_prefix": ""},
    "bge-m3": {"query_prefix": "", "doc_prefix": ""},
    "nomic-embed-text": {"query_prefix": "search_query: ", "doc_prefix": "search_document: "},
}

# --- Ingestion (01) -------------------------------------------------------
MIN_CHARS_FOR_TEXT_PAGE = 40
OCR_DPI = 200
OCR_IMAGE_MIN_AREA = 0.15
MIN_OCR_CHARS = 20
USE_VISION_OCR = True
VISION_CONF_THRESHOLD = 0.90
VISION_NUM_PREDICT = 2048                         # caps runaway vision output (Unsupervised p65 wrote 12k chars)

# --- Chunking (02) --------------------------------------------------------
CHUNKER_VERSION = "2"
MAX_TOKENS = 400
MIN_BODY_CHARS = 15
TITLE_SIZE_RATIO = 1.2
VISION_MAX_CHARS = 6000
DUP_RATIO = 0.95

# --- Retrieval (04) -------------------------------------------------------
RETRIEVAL_METHOD = "dense"                        # dense | bm25 | hybrid; dense won on the eval
K = 5
CANDIDATES = 20
RRF_K = 60
EMBED_BATCH = 16
RARE_TERM_MAX_DF = 3                              # a query word in at most this many chunks counts as rare (names, coined terms)

# --- Generation (05) ------------------------------------------------------
MAX_HISTORY_MESSAGES = 4
SUPPORT_MIN = 0.35
DENSE_FLOOR = 0.45                                # skip the LLM below this; lowest answerable question was 0.524
INSUFFICIENT = "INSUFFICIENT_CONTEXT"
ABSTAIN_MSG = "Your materials don't cover this."

# --- Router (06) ----------------------------------------------------------
K_BY_TYPE = {"definition": 4, "conceptual": 8, "followup": 5}
K_PER_SUBQUERY = 4
MAX_COMPARE_PASSAGES = 8

# --- Web (07) -------------------------------------------------------------
USER_AGENT = "EduRAG-student-project/0.1 (local study assistant)"
ALLOWLIST = ["docs.python.org", "wikipedia.org", "realpython.com", "geeksforgeeks.org", "w3schools.com",
             "programiz.com", "numpy.org", "pandas.pydata.org"]
DDGS_BACKENDS = ["duckduckgo", "bing", "google"]
N_SEARCH = 8
N_PAGES = 4
MAX_PAGE_CHARS = 20000
WEB_K = 5
WEB_MAX_TOKENS = 400

# --- Syllabus ---------------------------------------------------------------
SYLLABUS_MIN_SCORE = 0.52                         # question -> topic cosine; test syllabus: outside max 0.485, inside (by meaning) min 0.566
COVERAGE_MIN_SCORE = 0.56                         # topic -> slide cosine; test syllabus: real coverage >= 0.575, untaught topics 0.515 / 0.552

# --- Past papers (item 7) -------------------------------------------------
# Dense cosine, not the reranker: a cross-encoder scores "does this passage answer the question", and an exam
# question answers nothing (DBSCAN question vs "How does DBSCAN work?" = 0.0016). On a test paper matches scored
# 0.57-0.73 and non-matches 0.25-0.51.
PYQ_MIN_DENSE = 0.55
PYQ_MAX_HITS = 3


def chunk_settings():
    return {"version": CHUNKER_VERSION, "max_tokens": MAX_TOKENS, "min_body_chars": MIN_BODY_CHARS,
            "title_size_ratio": TITLE_SIZE_RATIO, "vision_max_chars": VISION_MAX_CHARS, "dup_ratio": DUP_RATIO}


def chunk_settings_hash(settings=None):
    """Same hash as notebook 02, so the existing embedding cache stays valid."""
    s = settings or chunk_settings()
    return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]


def ensure_dirs():
    for d in (LIBRARY_DIR, INDEX_DIR, CACHE_DIR, CHROMA_DIR, WEB_CACHE_DIR, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
