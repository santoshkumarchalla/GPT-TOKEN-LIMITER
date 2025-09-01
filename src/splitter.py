# src/splitter.py
from typing import List, Tuple
import re

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """
    Simple, robust chunker based on characters with sentence boundaries preference.
    chunk_size and overlap are measured in characters (not tokens) for portability.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        # Try to end at the last sentence boundary before end
        substring = text[start:end]
        last_period = substring.rfind('. ')
        last_newline = substring.rfind('\n')
        last_break = max(last_period, last_newline)
        if last_break > int(chunk_size * 0.3):
            end = start + last_break + 1  # include char after boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= text_len:
            break
    return chunks
