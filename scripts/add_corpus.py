"""Add every supported file in Corpus/ to the library (the same path as the app's upload button), smallest first.
Files already in the library (same name) are skipped, so the script can simply be run again after a power cut.
    python scripts/add_corpus.py
"""
import sys
import time
from pathlib import Path

from edu_rag import config, index, ingest, library

CORPUS = config.ROOT / "Corpus"
PAST_PAPERS = {"solved_python_exam.docx"}           # role pyq; everything else is lecture material


def main():
    have = {s["name"] for s in library.list_sources()}
    files = sorted((f for f in CORPUS.iterdir() if f.suffix.lower() in ingest.SUPPORTED_TYPES and f.name not in have),
                   key=lambda f: f.stat().st_size)
    print(f"{len(files)} files to add; {len(have)} already in the library", flush=True)
    t_all = time.time()
    for i, f in enumerate(files, 1):
        role = "pyq" if f.name in PAST_PAPERS else "textbook"
        t0 = time.time()
        try:
            doc, _ = index.add_file(f.name, f.read_bytes(), role)
            ocr = len(doc["ocr_pages"]) + len(doc["image_ocr_pages"])
            print(f"[{i}/{len(files)}] OK  {f.name}: {doc['n_pages']} pages, OCR {ocr}, vision {len(doc.get('vision_pages', []))}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as e:                      # one bad file must not stop the rest
            print(f"[{i}/{len(files)}] ERR {f.name}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
