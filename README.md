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

## Pipeline Overview
User query → Hybrid retrieval (FAISS dense + TF-IDF sparse) → Cross-encoder reranking → Evidence selection → Local LLM generation via Ollama (Gemma 2) → Safety check + final response.

## Example Prompts
## Example Prompts
Start with:
- "Hello, I’m Eva. My dog’s name is Max."

Then try:
- "My cat is vomiting 3 times today and seems tired. What could it be and what should I monitor?"
- "My dog has diarrhea for 2 days but is still eating. What home care steps are safe, and when should I see a vet?"
- "My cat is breathing fast and looks distressed. What should I do right now?"
- "My dog is scratching a lot and has red skin patches. What are possible causes and what questions should I answer?"
- "My cat stopped eating since yesterday and is hiding. What information do you need from me?"
> Note: You can start with a greeting (e.g., "Hello, I’m Eva") and optionally share your pet’s name (e.g., "My dog’s name is Max"). The assistant can remember the provided user/pet names during the session and use them in follow-ups.

## Sidebar Controls (Settings)
The app includes a Streamlit sidebar to configure generation and retrieval behavior.

### LLM
- **LLM model**: Select the local Ollama model (e.g., `gemma2:9b`).
- **Use LLM**: Toggle generation on/off (useful for testing retrieval-only behavior).
- **Temperature**: Controls randomness. Lower values (e.g., 0.2) produce more stable, conservative outputs.

### Retrieval
- **Top k chunks**: Number of evidence chunks passed forward after retrieval.
- **Candidate pool**: Size of the initial candidate set retrieved before reranking.
- **Min relevance**: Minimum similarity threshold to filter weak matches. Higher values reduce noise but may reduce coverage.

### Reranker
- **Use reranker**: Enables cross-encoder reranking to improve top evidence quality.
- **Rerank pool size**: Number of retrieved candidates reranked by the cross-encoder.
- **Show debug retrieval**: Displays retrieved / reranked evidence for inspection and evaluation.

### Safety
- **Show safety alerts**: Displays rule-based urgent-care warnings when critical cues are detected.

### Nearby Vets (Prototype)
- **Your location**: Best results with `city + district` (and country if needed).
- **Search radius (km)**: Controls the search radius for nearby clinics.
- **Show nearby veterinary clinics (every answer)**: If enabled, the app tries to show clinics on each response (may increase latency).
- **Auto-show vets on urgent alerts**: If enabled, the app triggers vet suggestions when an urgent safety alert is raised.

Note: The Nearby Vets feature is a prototype and may be slow or unreliable depending on network conditions and location signal.

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
