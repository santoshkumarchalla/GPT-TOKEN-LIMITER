#!/bin/bash
# Example usage:
# 1) create .env from .env.example and set OPENAI_API_KEY
# 2) ingest a file

python -m src.app --file examples/thinkAndGrowRich.txt --doc-id sample_doc
# 3) query
python -m src.query --q "Summarize the First Chapter described in the document" --topk 8
