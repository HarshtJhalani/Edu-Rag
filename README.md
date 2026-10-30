# Edu-RAG: An Adaptive RAG Framework for Educational Question Answering

A local study assistant. Students add their own course material (lecture slides, notes, past papers, a syllabus)
and ask questions. Answers come **only from that material**, with the page or slide each fact came from.
Everything runs on the student's own computer with [Ollama](https://ollama.com): no cloud APIs, no cost,
and nothing leaves the machine unless the student presses **Go deeper**.

## Features

| | |
|---|---|
| **Grounded answers** | Every claim cites a passage (`[S1]`); page numbers come from metadata, never from the model. Answers stream live. |
| **Slide previews** | Each citation opens the real slide/page with the cited text highlighted. |
| **Adaptive router** | Classifies each message (definition, explanation, comparison, follow-up, small talk) and picks the retrieval strategy: compare questions are split into sub-queries, follow-ups are rewritten using the chat history. |
| **Honest abstaining** | "Your materials don't cover this." when the model finds the passages insufficient or the question is off-topic. |
| **Go deeper (opt-in)** | Searches Wikipedia/DuckDuckGo, prefers trusted sites, answers with `[W#]` links and a "confirmed by 2 sources" badge, kept separate from the grounded answer. Pages can be added to the library. |
| **Flashcards** | Cards written from the slides, reviewed with spaced repetition (Leitner boxes: known cards come back after 1, 2, 4, 8, 16 days; missed ones the same day). |
| **Exam mode** | A timed mock paper: past-paper questions first (optionally only chosen syllabus units), topped up with exam-style questions from the slides. Auto-submits when time runs out; each answer is marked out of its marks with feedback and a model answer citing the slides. |
| **Question-paper generator** | Learns the pattern of the student's previous-year papers (sections, marks, "attempt any N"), charts which topics they ask most, and writes a NEW paper in that pattern from the slides, weighted towards frequent topics. PDF paper, PDF answer key citing slide pages, or a timed attempt in Exam mode. |
| **Practice quiz** | Multiple-choice and short-answer questions written from the student's slides, marked with feedback and the source slide. |
| **Cheat sheet** | Revision notes for a deck or topic, every point with its page; download as Markdown or PDF. |
| **Syllabus scope** | Units and topics read from the syllabus; questions tagged with their unit or flagged "outside your syllabus"; coverage view shows which topics have no material yet. |
| **Past papers** | Shows when a topic was asked in a previous-year paper, with the paper and page. |
| **Answer tools** | Copy an answer with its sources, rate it 👍/👎 (saved locally), and from any cited slide: "Explain this slide" or "Flashcards from it". The Library shows the questions asked most about each source. |
| **Chat history + export** | Earlier chats of the session in the sidebar; download a chat as Markdown or PDF with every cited page. |
| **Light / dark theme** | Light, Dark or Auto (follows Windows) from the sidebar, switched live and remembered. |
| **Many formats** | PDF (text layer + OCR for scans and screenshots), PowerPoint, Word, Jupyter notebooks, pasted notes. |

## Requirements

- Windows 10/11 (tested), Python 3.11+ (developed on 3.14)
- [Ollama](https://ollama.com) (the app offers to install it with `winget` on first launch)
- An NVIDIA GPU with 8 GB VRAM is recommended (the app picks a 3B model on smaller GPUs)
- Optional: Microsoft PowerPoint / Word, for slide previews of `.pptx` / `.docx` files

Models (downloaded once, the app shows the sizes and asks first):

| Role | Model |
|---|---|
| Answers, router, quiz | `qwen2.5:7b-instruct` (or `qwen2.5:3b-instruct` under 7 GB VRAM) |
| Embeddings | `bge-m3` |
| OCR for screenshots and scanned slides | `qwen3-vl:4b-instruct-q4_K_M` (RapidOCR handles clean text first) |

## Setup

```bash
git clone <this repo>
cd Edu-Rag
python -m venv rp
rp\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run

Double-click **`start.bat`**, or:

```bash
rp\Scripts\python.exe -m streamlit run app.py
```

On first launch the setup screen checks Ollama, the GPU and the models. Then open **📚 Library** and add your
slides (and optionally past papers and a syllabus). Course material is **not** part of this repository.

## Results

Evaluated on 68 questions written from 10 lecture decks (52 answerable, 16 not covered by the slides;
28 of them reworded to avoid slide wording). Questions and all results are in [`data/eval/`](data/eval).

| What | Result |
|---|---|
| Best retrieval (04) | dense search with `bge-m3`: nDCG@5 0.809, Recall@5 0.880 (beat nomic-embed-text, qwen3-embedding, BM25 and hybrid RRF) |
| Full corpus (51 sources, ~2,600 passages) | hybrid BM25 + dense beats dense: gold slide found 0.904 → 0.942, hard-question recall 0.739 → 0.784, cites a gold page 0.86–0.88 → 0.904, 52/52 answered, same 2.3 s; the app now uses hybrid |
| Reranker reordering (04) | hurt every dense configuration (nDCG@5 0.809 → 0.724), so it is off |
| Answerable questions answered | 52 / 52 |
| Unanswerable questions refused | 15 / 16 |
| Answers with valid citations | 52 / 52 |
| Answers citing a correct page | 94 % |
| Router vs fixed retrieval (06) | page recall 0.88 → 0.92 (hard questions 0.77 → 0.84; comparisons 0.85 → 1.00) |
| Follow-up questions (06) | correct slide found 9/10 with the router's rewrite vs 7/10 without |
| Answer time | about 2-3 s on an RTX 3070 Ti |

Design decisions the data changed (details in the notebooks):
- **Abstaining**: the cross-encoder reranker score refused 14 of 22 reworded answerable questions; the LLM's own
  "insufficient context" signal separated answerable from unanswerable questions perfectly on this set, so it decides.
- **Coined names in notes** ("Meow Laadle") are missed by dense embeddings; chunks sharing the question's rare words
  are added first.
- **Past-paper matching** uses dense similarity: the reranker scored a matching exam question at 0.0016, because an
  exam question does not *answer* anything.
- Timings in notebooks 04-07 include a ~2 s per-request overhead from resolving `localhost` on Windows; the app uses
  `127.0.0.1` (see `config.py`).

## Project layout

```
app.py                 Streamlit entry point: setup check + navigation
app_pages/             UI pages: chat, flashcards, quiz, exam, question paper, cheat sheet, syllabus, library
assets/                logo and icon
src/edu_rag/           the system (plain functions, no classes)
  config.py            every setting in one place
  ingest.py, readers.py      PDF + OCR, PowerPoint, Word, Jupyter, pasted text
  chunking.py          one slide = one chunk; code and tables never split
  library.py           data/library/<role>/<source>/ folders
  index.py             embedding cache, Chroma, BM25; sync() keeps them in line with the library
  retrieve.py          dense / BM25 / RRF hybrid (default), filters, rare-word match, past papers
  generate.py          prompt, citation checking, support check, abstaining
  router.py            the adaptive router
  web.py               Go deeper
  previews.py          highlighted page images
  cheatsheet.py, quiz.py, syllabus.py
  flashcards.py        cards + Leitner spaced repetition (deck in data/study/)
  exam.py              mock papers from past papers + slides, examiner-style marking
  paper.py             question-paper generator: pattern, topic frequency, new paper, answer key
  activity.py          asked questions and answer ratings (data/study/)
  export.py            chat -> Markdown / PDF, shared HTML -> PDF writer
  setup_check.py       first-launch checks
notebooks/             01-07: prototypes and evaluations (historical; written before data/ was reorganised)
tests/                 pytest, no Ollama needed
scripts/               run_eval.py (evaluation), add_corpus.py + corpus_run.bat (bulk import), migrate_library.py
data/eval/             evaluation questions and results
```

Created at runtime and not in the repository: `data/library/` (your material), `data/study/` (your flashcard
progress), `data/index/` and `data/cache/` (rebuilt automatically). `data/eval/sample_papers/` holds two SAMPLE past papers written for testing the paper
generator (not real university papers; PDFs are not committed).

## Tests

```bash
rp\Scripts\python.exe -m pytest
```

The 68-question evaluation runs through the app's own pipeline on whatever is in the library:

```bash
rp\Scripts\python.exe scripts\run_eval.py my_run hybrid
```

The last argument picks the retrieval method (`dense`, `bm25` or `hybrid`; default: `RETRIEVAL_METHOD` in
`config.py`, which is `hybrid`). Results are written to `data/eval/my_run.json`.

## Limitations and future work

- Thresholds (syllabus scope, coverage, past papers) were tuned on small test sets; check them on real syllabi and papers.
- The unanswerable set has 16 questions; the "insufficient context" signal needs a larger test.
- Memory lasts within one chat (last 4 messages + follow-up rewriting); persistent memory across sessions and a
  learner profile of weak topics are future work.
- Exam marking by the 7B model is strict: complete answers often get 60-80 % of the marks. Use the feedback and model
  answer, not the score alone. Ranking is sensible (wrong or blank answers get 0).
- The paper generator was tested on two sample papers written for testing, not on real university papers. With few past
  papers, topic frequencies are small counts; questions are checked against their slide (>= 30 % of their content words).
- Chat history lasts until the app is closed (export a chat to keep it).
- OCR occasionally misreads decorative slide layouts; the vision model's output is capped to stop runaway transcriptions.

## License

Code: MIT, see [LICENSE](LICENSE). Course material used during development (lecture slides, papers) belongs to its
authors and is not included in this repository.
