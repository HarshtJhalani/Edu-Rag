@echo off
rem Baseline eval -> add all of Corpus/ -> eval again. Log: data\eval\corpus_run.log
cd /d "%~dp0.."
set "PYTHONIOENCODING=utf-8"
if not exist data\eval\eval_before_corpus.json rp\Scripts\python.exe scripts\run_eval.py eval_before_corpus
rp\Scripts\python.exe scripts\add_corpus.py
rp\Scripts\python.exe scripts\run_eval.py eval_after_corpus
echo ALL DONE
