# src/app.py
"""
CLI ingest script:
    python -m src.app --file /path/to/huge.txt --doc-id mydoc1
It will:
    - chunk
    - embed with sentence-transformers
    - add to FAISS and sqlite metadata
    - checkpoint progress to data/checkpoints.json so the operation can resume
"""
import argparse
import os
from src.splitter import chunk_text
from src.embedder import Embedder
from src.store import VectorStore, load_checkpoints, save_checkpoints
from src.utils import read_file
import numpy as np
from tqdm import tqdm

def ingest(file_path: str, doc_id: str, chunk_size: int = 2000, overlap: int = 200, batch_size: int = 64):
    text = read_file(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"Total chunks: {len(chunks)}")

    # Load checkpoints
    checkpoints = load_checkpoints()
    last_index = checkpoints.get(doc_id, -1)
    start_index = last_index + 1
    print(f"Resuming from chunk index: {start_index}")

    embedder = Embedder()
    dim = embedder.model.get_sentence_embedding_dimension()
    store = VectorStore(dim=dim)

    for i in range(start_index, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        embs = embedder.embed_texts(batch_chunks)
        metas = []
        for j, c in enumerate(batch_chunks):
            metas.append({'doc_id': doc_id, 'chunk_index': i+j, 'text': c})
        n_before, n_after = store.add(embeddings=np.array(embs), metadatas=metas)
        print(f"Stored chunks {i}..{i+len(batch_chunks)-1} -> total vectors: {n_after}")
        # checkpoint last processed index
        checkpoints[doc_id] = i + len(batch_chunks) - 1
        save_checkpoints(checkpoints)

    print("Ingest complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to input text file")
    parser.add_argument("--doc-id", required=True, help="Document id label")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    ingest(args.file, args.doc_id, chunk_size=args.chunk_size, overlap=args.overlap, batch_size=args.batch_size)
