# src/query.py
"""
Query CLI:
    python -m src.query --q "Explain the architecture around X" --topk 8
This will:
    - embed the query locally
    - retrieve top-k chunks
    - optionally create a summarized context that fits the token budget
    - call OpenAI ChatCompletion with the assembled context and user's question
"""
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

from src.embedder import Embedder
from src.store import VectorStore
from src.retriever import Retriever
from src.summarizer import iterative_summarize, call_chatgpt
from src.utils import estimate_tokens_by_chars

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CHAT_CONTEXT_TOKENS = int(os.getenv("MAX_CHAT_CONTEXT_TOKENS", "3500"))

def build_context_text(results):
    # results: list of {score, doc_id, chunk_index, text}
    # sort by score desc
    results_sorted = sorted(results, key=lambda r: r['score'], reverse=True)
    texts = []
    for r in results_sorted:
        header = f"[doc:{r['doc_id']} chunk:{r['chunk_index']} score:{r['score']:.3f}]"
        texts.append(header + "\n" + r['text'])
    return texts

def query(query_str: str, topk: int = 8):
    embedder = Embedder()
    dim = embedder.model.get_sentence_embedding_dimension()
    store = VectorStore(dim=dim)
    retriever = Retriever(store, embedder)
    results = retriever.retrieve(query_str, top_k=topk)
    if not results:
        print("No matching chunks found.")
        return

    # assemble context; first try directly
    texts = build_context_text(results)
    joined = "\n\n".join(texts)
    estimated = estimate_tokens_by_chars(joined)
    print(f"Retrieved {len(results)} chunks, estimated tokens: {estimated}")

    # If too large, run iterative summarizer
    if estimated > MAX_CHAT_CONTEXT_TOKENS:
        print("Context too big — summarizing retrieved chunks to fit token budget...")
        summary = iterative_summarize(texts, target_token_budget=MAX_CHAT_CONTEXT_TOKENS//2)
        context_for_model = summary
    else:
        context_for_model = joined

    # Now call ChatGPT with RAG prompt
    system = "You are an assistant that answers using the provided context. If the answer is not found in the context, say 'Not in provided documents'."
    user_prompt = f"Context:\n{context_for_model}\n\nUser question:\n{query_str}\n\nAnswer using only the context above; keep the answer concise and cite doc ids where appropriate."

    # final call
    answer = call_chatgpt(system, user_prompt, model=MODEL, max_tokens=800)
    print("\n===== ANSWER =====\n")
    print(answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    query(args.q, topk=args.topk)
