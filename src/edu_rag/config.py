"""All settings in one place. Plain module constants; values chosen in notebooks 02-07."""
import hashlib
import json
import os
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
APP_SETTINGS_FILE = DATA_DIR / "app_settings.json"
STUDY_DIR = DATA_DIR / "study"                    # the student's own progress: flashcard boxes
FLASHCARDS_FILE = STUDY_DIR / "flashcards.json"
QUESTIONS_LOG = STUDY_DIR / "questions.jsonl"     # asked questions + cited sources ("most asked" in the Library)
FEEDBACK_LOG = STUDY_DIR / "feedback.jsonl"       # thumbs up/down on answers

ROLES = ["textbook", "note", "web", "pyq", "syllabus"]
EVIDENCE_ROLES = ["textbook", "note", "web"]      # only these may support an answer

# --- Ollama ---------------------------------------------------------------
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and Ollama listens on IPv4 only, which added
# ~2 s to EVERY request (Ollama reported 0.28 s for an embedding the client waited 2.4 s for).
OLLAMA_URL = "http://127.0.0.1:11434"
CPU_FORCED = os.environ.get("EDU_RAG_CPU") == "1"  # this variable keeps the whole run on the processor
CPU_MODE = CPU_FORCED                              # running on the processor, by choice or for lack of a GPU; set by setup_check
CPU_ONLY = CPU_FORCED                              # keep models off the GPU; the sidebar toggle changes it through setup_check.use_cpu
LLM = "qwen2.5:7b-instruct"
LLM_SMALL = "qwen2.5:3b-instruct"                 # picked by setup_check when VRAM is under 7 GB
EMBED_MODEL = "bge-m3"                            # best on the 68-question eval (04)
VISION_MODEL = "qwen3-vl:4b-instruct-q4_K_M"
RERANKER = "BAAI/bge-reranker-base"               # CPU; evaluation only (04/05), not used by the app
ANSWER_NUM_PREDICT = 1500                         # 3B once wrote 53,348 words for one answer; the longest real answers (12 marks) are about 300 words
ANSWER_CUT_NOTE = "(This answer was cut off because it was running too long or repeating itself. Try asking again.)"
LOOP_TAIL_CHARS = 80                              # the same last 80 characters written 3 times over: the model is stuck, stop it
LOOP_REPEATS = 3
JSON_NUM_PREDICT = 2048                           # 3B in JSON mode can write without end; a request would wait out its 10-minute timeout
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
# dense | bm25 | hybrid. Dense won on the 532-passage library (04); on the full 2,600-passage corpus hybrid (BM25 + dense,
# RRF) found the gold slide more often (hit 0.904 -> 0.942, hard recall 0.739 -> 0.784, all 52 answered) at the same speed.
RETRIEVAL_METHOD = "hybrid"
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
K_LIST = 8                                        # "types of X": the kinds of X are usually on separate slides
FASTER_MAX_PASSAGES = 5                           # 3B in CPU mode: reading the passages is most of the wait (21 s -> 15 s per answer)
MAX_COMPARE_PASSAGES = 8
# Keyboard mashes ("qwertyuiasdfghjkl") never reach the router LLM, which once rewrote one, using the chat history,
# into a question about the previous answer. A message is unclear when none of its words occur in the materials AND its
# letters look unlike them: mean character-trigram log-prob (model trained on the corpus words) below this. Real questions,
# also about topics not in the materials ("what is kubernetes"), scored -2.2 to -3.5; keyboard mashes -3.6 to -4.4.
GIBBERISH_MAX_SCORE = -3.6
GIBBERISH_MIN_LEN = 4                             # shorter words (acronyms: bfs, dfs, sql) are not judged

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
SYLLABUS_MIN_SCORE = 0.53                         # question -> topic cosine; test syllabus: outside max 0.485, inside (by meaning) min 0.566
COVERAGE_MIN_SCORE = 0.56                         # topic -> slide cosine; test syllabus: real coverage >= 0.575, untaught topics 0.515 / 0.552

# --- Past papers (item 7) -------------------------------------------------
# Dense cosine, not the reranker: a cross-encoder scores "does this passage answer the question", and an exam
# question answers nothing (DBSCAN question vs "How does DBSCAN work?" = 0.0016). On a test paper matches scored
# 0.57-0.73 and non-matches 0.25-0.51.
PYQ_MIN_DENSE = 0.55
PYQ_MAX_HITS = 3

# --- Flashcards / exam ----------------------------------------------------------
LEITNER_DAYS = [0, 1, 2, 4, 8, 16]                # box -> days until the card is due again; "knew it" moves a card up a box
EXAM_DEFAULT_MARKS = 5                            # a past-paper question with no marks printed on it
EXAM_MARK_K = 5                                   # slide passages the marker gets per question


def chunk_settings():
    return {"version": CHUNKER_VERSION, "max_tokens": MAX_TOKENS, "min_body_chars": MIN_BODY_CHARS,
            "title_size_ratio": TITLE_SIZE_RATIO, "vision_max_chars": VISION_MAX_CHARS, "dup_ratio": DUP_RATIO}


def chunk_settings_hash(settings=None):
    """Same hash as notebook 02, so the existing embedding cache stays valid."""
    s = settings or chunk_settings()
    return hashlib.sha1(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]


def ensure_dirs():
    for d in (LIBRARY_DIR, INDEX_DIR, CACHE_DIR, CHROMA_DIR, WEB_CACHE_DIR, STUDY_DIR):
        d.mkdir(parents=True, exist_ok=True)
