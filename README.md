# FastAPI microservice exposing AI endpoints

- **FastAPI microservice** exposing AI endpoints
  - `/health` -- service and index readiness
  - `/classify` -- categorize text into \[bug, feature_request,
    praise, question\]
  - `/rag` -- retrieval‑augmented QA using FAISS + local embeddings
  - `/reload-index` -- rebuild after adding docs
- **RAG pipeline**
  - Load `.txt` files from `./data`
  - Chunk text with overlap
  - Encode with SentenceTransformers
  - Store/retrieve with FAISS
- **Generation**
  - Default: [Ollama](https://ollama.ai) local models (configurable
    with env vars)
  - Designed for zero external API cost

## Quickstart

```bash
git clone https://github.com/AshwinSathian/prompting-basics.git
cd prompting-basics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run
uvicorn main:app --reload
```

Visit <http://127.0.0.1:8000/docs>.

## Example Usage

```bash
# health
curl -s http://127.0.0.1:8000/health | jq

# classify
curl -s -X POST http://127.0.0.1:8000/classify   -H "Content-Type: application/json"   -d '{"text": "The app crashes on login"}'

# rag
curl -s -X POST http://127.0.0.1:8000/rag   -H "Content-Type: application/json"   -d '{"question": "Explain RAG"}'
```

## Repo Structure

- `main.py` -- FastAPI app
- `data/` -- text files to be indexed
- `requirements.txt` -- dependencies

## Environment Variables

- `EMBED_MODEL_NAME` -- embedding model (default: all-MiniLM-L6-v2)
- `OLLAMA_HOST` -- Ollama server (default: http://localhost:11434)
- `OLLAMA_MODEL` -- generation model (default: llama3)

## License

MIT © Ashwin Sathian
