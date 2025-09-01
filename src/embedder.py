# src/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Compact, fast, good for retrieval
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Accepts list of strings, returns 2D numpy array of embeddings.
        """
        embs = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Normalize
        norms = (embs ** 2).sum(axis=1, keepdims=True) ** 0.5
        norms[norms == 0] = 1.0
        return embs / norms
