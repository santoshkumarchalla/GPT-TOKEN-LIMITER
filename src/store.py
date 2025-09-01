# src/store.py
import faiss
import numpy as np
import os
import json
import sqlite3
from typing import List, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(DATA_DIR, 'checkpoints.json')
META_DB = os.path.join(DATA_DIR, 'metadata.db')
INDEX_FILE = os.path.join(DATA_DIR, 'faiss.index')

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self._load_or_init_index()
        self._init_meta_db()

    def _init_meta_db(self):
        self.conn = sqlite3.connect(META_DB)
        cur = self.conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                doc_id TEXT,
                chunk_index INTEGER,
                text TEXT
            )
        ''')
        self.conn.commit()

    def _load_or_init_index(self):
        if os.path.exists(INDEX_FILE):
            try:
                self.index = faiss.read_index(INDEX_FILE)
            except Exception:
                self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def save_index(self):
        faiss.write_index(self.index, INDEX_FILE)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """
        embeddings: shape (N, dim)
        metadatas: list with keys doc_id, chunk_index, text
        we append rows to sqlite and add vectors to faiss
        """
        n_before = self.index.ntotal
        # FAISS expects float32
        arr = embeddings.astype('float32')
        self.index.add(arr)
        # Add metadata rows
        cur = self.conn.cursor()
        for meta in metadatas:
            cur.execute('''
                INSERT INTO chunks (doc_id, chunk_index, text)
                VALUES (?, ?, ?)
            ''', (meta['doc_id'], meta['chunk_index'], meta['text']))
        self.conn.commit()
        self.save_index()
        return n_before, self.index.ntotal

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        """
        query_emb: shape (dim,) or (1, dim)
        returns list of (score, metadata_row)
        """
        q = query_emb.astype('float32').reshape(1, -1)
        D, I = self.index.search(q, top_k)
        ids = I[0].tolist()
        scores = D[0].tolist()
        results = []
        cur = self.conn.cursor()
        for idx, score in zip(ids, scores):
            if idx < 0:
                continue
            # SQLite rows are 1-based inserted IDs; align by rowid
            cur.execute('SELECT doc_id, chunk_index, text FROM chunks WHERE id = ?', (idx,))
            row = cur.fetchone()
            if row:
                results.append({'score': float(score), 'doc_id': row[0], 'chunk_index': row[1], 'text': row[2]})
        return results

    def get_total_vectors(self):
        return int(self.index.ntotal)

# Checkpoint helpers
def load_checkpoints():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoints(obj):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
