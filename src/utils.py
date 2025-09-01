# src/utils.py
import os
import math

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def estimate_tokens_by_chars(text: str) -> int:
    # rough heuristic: 4 chars per token
    return max(1, int(len(text) / 4))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
