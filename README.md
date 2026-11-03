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
| **Difficulty levels** | Easy (recall), Medium (understanding) or Hard (application and analysis) in the quiz, Exam mode and the question paper. Questions of the wrong kind are rejected in code and written again. In Exam mode the level also sets the marks (2 / 5 / 10) and picks past-paper questions by their marks; a generated paper keeps its past paper's marks. |
| **Cheat sheet** | Revision notes for a deck or topic, every point with its page; download as Markdown or PDF. |
| **Syllabus scope** | The whole syllabus is read course by course into semester, course, unit and topics, with a semester selector (when it names semesters) shared by the Syllabus and Exam pages; questions tagged with their unit when one matches; coverage view shows which topics have no material yet. |
| **Past papers** | Shows when a topic was asked in a previous-year paper, with the paper and page. |
| **Answer tools** | Copy an answer with its sources, rate it thumbs up or down (saved locally), and from any cited slide: "Explain this slide" or "Flashcards from it". The Library shows the questions asked most about each source. |
| **Chat history + export** | Earlier chats of the session in the sidebar; download a chat as Markdown or PDF with every cited page. |
| **Light / dark theme** | Light, Dark or Auto (follows Windows) from the sidebar, switched live and remembered. |
| **Many formats** | PDF (text layer + OCR for scans and screenshots), PowerPoint, Word, Jupyter notebooks, pasted notes. |

## Requirements

- Windows 10/11 (tested), Python 3.11+ (developed on 3.14)
- [Ollama](https://ollama.com) (the app offers to install it with `winget` on first launch)
- An NVIDIA GPU with 8 GB VRAM is recommended (the app picks a 3B model on smaller GPUs)
- No GPU? It still runs: see [Running without a GPU](#running-without-a-gpu)
- Optional: Microsoft PowerPoint / Word, for slide previews of `.pptx` / `.docx` files

Models (downloaded once, the app shows the sizes and asks first):

| Role | Model |
|---|---|
| Answers, router, quiz | `qwen2.5:7b-instruct` (or `qwen2.5:3b-instruct` under 7 GB VRAM) |
| Embeddings | `bge-m3` |
| OCR for screenshots and scanned slides | `qwen3-vl:4b-instruct-q4_K_M` (RapidOCR handles clean text first) |

## Running without a GPU

With no NVIDIA GPU the app switches to CPU mode by itself: the smaller `qwen2.5:3b-instruct` answer model, and no vision model
(about 3 minutes per page on a CPU), so only 3.1 GB is downloaded. At least 8 GB of RAM is recommended.

Measured on a Ryzen 7 3700X (8 cores) with the models forced off the GPU:

| | CPU | RTX 3070 Ti |
|---|---|---|
| Answer, `qwen2.5:3b-instruct` (at most 5 passages) | 17-32 s; worked examples 19-86 s | not the default |
| Answer, `qwen2.5:7b-instruct` | 16-73 s; worked examples 48-143 s | about 2.3 s |
| Quiz of 4 questions, 3B / 7B | 40 s (before the 3B fix, which made no questions) / 148 s | |
| Flashcards (4), cheat sheet, exam paper (3), 7B | 44 s, 69 s, 57 s | |
| Vision OCR of one page | about 174 s (off in CPU mode) | about 8 s |

In CPU mode the sidebar has an **Answer model** switch. **Faster** is the default; **More accurate** uses `qwen2.5:7b-instruct`, needs at least 12 GB of RAM (7B and the embedding model take about 6.2 GB) and asks before its 4.7 GB download. With less RAM the switch is not offered.

Most of the wait is the model reading the passages before its first word, so Faster sends at most 5 (`config.FASTER_MAX_PASSAGES`) instead of up to 8. More passages cite better: on the 68-question evaluation (one run each), Faster cites a correct page on 40 of 52 answerable questions (77%), 3B with up to 8 passages on 44, and 7B on 47 (90%). All three answer all 52. Faster refuses 5 of the 16 uncovered questions, 7B refuses 7.

Scanned pages are read with RapidOCR only in CPU mode, so code screenshots and tables come out less accurately; a scanner's text
layer is still used. On a machine with a GPU, the sidebar's **Run on** toggle moves between the GPU and the processor without a restart. Setting `EDU_RAG_CPU=1` before starting the app keeps the whole run on the processor and hides the toggle.

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

On first launch the setup screen checks Ollama, the GPU and the models. Then open **Library** and add your
slides (and optionally past papers and a syllabus). Course material is **not** part of this repository.

## Results

Evaluated on 68 questions written from 10 lecture decks (52 answerable, 16 not covered by the slides;
28 of them reworded to avoid slide wording). The full evaluation is described in the project report.

| What | Result |
|---|---|
| Best retrieval (04) | dense search with `bge-m3`: nDCG@5 0.809, Recall@5 0.880 (beat nomic-embed-text, qwen3-embedding, BM25 and hybrid RRF) |
| Full corpus (51 sources, ~2,600 passages) | hybrid BM25 + dense beats dense: gold slide found 0.904 → 0.942, hard-question recall 0.739 → 0.784, cites a gold page 0.86–0.88 → 0.904, 52/52 answered, same 2.3 s; the app now uses hybrid |
| Reranker reordering (04) | hurt every dense configuration (nDCG@5 0.809 → 0.724), so it is off |
| Answerable questions answered | 52 / 52 |
| Questions the materials do not cover, refused | 7 / 16 on the full library |
| Answers with valid citations | 52 / 52 |
| Answers citing a correct page | about 90 % |
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
scripts/               add_corpus.py (bulk import), migrate_library.py
```

Created at runtime and not in the repository: `data/library/` (your material), `data/study/` (your flashcard
progress), `data/index/` and `data/cache/` (rebuilt automatically).

## Limitations and future work

- Thresholds (syllabus scope, coverage, past papers) were tuned on small test sets; check them on real syllabi and papers.
- The unanswerable set has 16 questions; the "insufficient context" signal needs a larger test.
- Memory lasts within one chat (last 4 messages + follow-up rewriting); persistent memory across sessions and a
  learner profile of weak topics are future work.
- Exam marking by the 7B model is strict: complete answers often get 60-80 % of the marks. Use the feedback and model
  answer, not the score alone. Ranking is sensible (wrong or blank answers get 0).
- The paper generator was tested on two sample papers written for testing, not on real university papers. With few past
  papers, topic frequencies are small counts; questions are checked against their slide (>= 30 % of their content words).
- Difficulty levels are checked by question form, not validated against students or lecturers. Hard questions need the
  model to reason, so check surprising quiz answers against the cited slide: hard multiple-choice questions are dropped
  when the model's second, independent answer disagrees, but that does not catch every wrong key. A hard paper can also
  ask too much for a 2-mark slot.
- Chat history lasts until the app is closed (export a chat to keep it).
- A scanned syllabus uses the scanner's text with the vision model only on unreadable pages: fast, but course titles can keep scan typos, and unusually laid-out lists can be missed (one practical course on the test syllabus).
- OCR occasionally misreads decorative slide layouts; the vision model's output is capped to stop runaway transcriptions.

## License

Code: MIT, see [LICENSE](LICENSE). Course material used during development (lecture slides, papers) belongs to its
authors and is not included in this repository.
