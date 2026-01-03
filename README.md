# PetSymptomChatbot-NLP2025

Pet Symptom Chatbot: evidence grounded symptom support using hybrid RAG (FAISS + TF-IDF) with cross-encoder reranking and local LLM generation via Ollama (Gemma 2). Built with Streamlit.

## Key Features
- Hybrid retrieval: dense (FAISS) + sparse (TF-IDF)
- Cross-encoder reranking for higher-precision evidence selection
- Evidence-grounded responses (supportive guidance, not a diagnosis)
- Basic safety / emergency cue detection (rule-based)
- Optional vet locator module (prototype; may be unstable depending on network / location signal)

## Repository Structure
- `app/` Streamlit UI (`chat_app.py`)
- `src/` retrieval, reranking, indexing, scraping / corpus utilities
- `data/` input corpora (CSV/JSONL) used to build the knowledge base
- `models/` generated artifacts (not tracked): FAISS index, TF-IDF matrix, embeddings, metadata

## Requirements
- Python 3.10+ (recommended)
- Streamlit
- Ollama installed and running locally (for local LLM generation)

## Install
```bash
pip install -r requirements.txt
```

## Build the Knowledge Base (Generate `models/`)
Run:
```bash
python src/build_embeddings.py
python src/build_index.py
```
### Ollama model (Gemma 2)
This project uses a local Ollama model (Gemma 2). Ensure it is available locally before running the app.
Example:
```bash
ollama pull gemma2
```
Expected outputs in `models/` after building:
- `chunks.pkl` (chunked documents / passages)
- `meta.pkl` (metadata per chunk)
- `titles.pkl` (titles per source / chunk)
- `urls.pkl` (source URLs)
- `doc_embeddings.npy` (dense embeddings)
- `faiss_index.bin` (FAISS vector index)
- `tfidf_vectorizer.pkl` (TF-IDF vectorizer)
- `tfidf_matrix.npz` (TF-IDF sparse matrix)

## Run the App
```bash
streamlit run app/chat_app.py
```
## Optional: Download prebuilt `models/`
If you do not want to build indexes locally, download `models.zip` from the Releases page, unzip it, and place the `models/` folder at the repository root.

## Notes on Safety and Scope
This project provides evidence-grounded informational support and basic triage-style guidance. It is not a veterinary diagnosis tool. If symptoms are severe (e.g., breathing distress, collapse, uncontrolled bleeding, seizures), users should seek urgent veterinary care.

## Known Limitations
- Retrieval quality depends on corpus coverage and chunking; vague queries may retrieve noisy evidence
- Indexing artifacts must be generated locally before running the app
- Vet locator module may be slow or fail due to rerun behavior, network restrictions, or missing location signals

## Future Improvements
- Stronger sparse ranking (BM25-style) and improved reranking strategies
- Better multi-turn grounding (dialogue state tracking to reduce repetition)
- More robust safety detection for subtle emergency cues
- More reliable vet search via explicit user-triggered lookup + caching, or a verified places API
