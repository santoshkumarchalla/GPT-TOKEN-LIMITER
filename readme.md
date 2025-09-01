📁 TOKEN-LIMITER/
├─ 📄 README.md
├─ 📄 requirements.txt
├─ 📄 .env.example
├─ 📁 data/
│  ├─ 📄 checkpoints.json
│  └─ 📄 metadata.db
├─ 📁 src/
│  ├─ 📄 app.py                 # CLI to ingest files & build vectors
│  ├─ 📄 query.py               # Query interface to send retrieval-augmented prompts to ChatGPT
│  ├─ 📄 splitter.py            # Chunking logic with overlap
│  ├─ 📄 embedder.py            # Embedding using sentence-transformers
│  ├─ 📄 store.py               # FAISS vector store + sqlite metadata + checkpointing
│  ├─ 📄 retriever.py           # Retrieve top-k chunks and assemble context
│  ├─ 📄 summarizer.py          # Iterative summarization to fit token budget (uses OpenAI)
│  └─ 📄 utils.py               # helper functions (token estimation, progress bars)
└─ 📁 examples/
   ├─ 📄 sample_long_text.txt
   └─ 📄 run_example.sh



- Create and Activate a Virtual Environment
    Create a virtual environment: python -m venv .venv
    Activate the virtual environment:
        For Windows: .venv\Scripts\activate
        For macOS/Linux: source .venv/bin/activate
- Install Requirements
    Install all the requirements given in requirements.txt by running the command: pip install -r requirements.txt

