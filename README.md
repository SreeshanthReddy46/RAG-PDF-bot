# PDF Document Assistant

A small Python app that reads PDF files and lets you ask questions about their contents. It uses Google Gemini to create answers and embeddings, and ChromaDB to store and search document text.

## Setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (optional but recommended).
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your Google AI API key:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

The app creates its upload and database folders under `data/` automatically.

## Add a PDF

Index a PDF before asking questions:

```bash
python cli.py upload path/to/document.pdf
```

The PDF is copied into `data/uploads/` and its text is indexed in the local Chroma database. Uploading a PDF with the same filename replaces its previous indexed content.

## Chat

Choose one of these interfaces:

```bash
python cli.py chat
```

```bash
streamlit run web_chat.py
```

```bash
uvicorn src.server:app --reload
```

The API accepts `POST /api/chat` with JSON such as:

```json
{"question": "What is this document about?", "session_id": "my-session"}
```

It returns an answer as JSON. Chat history is kept in memory for each session while the app is running.

## Project files

- `cli.py` — upload PDFs and chat in the terminal.
- `web_chat.py` — Streamlit chat interface.
- `src/server.py` — FastAPI chat endpoint.
- `src/ingestion.py` — extract, split, and index PDF text.
- `src/rag_engine.py` — retrieve relevant text and generate answers.
- `src/config.py` — API key, model, and storage settings.
- `data/` — uploaded PDFs and local vector database.

Answers are based on the text found in indexed PDFs. If that text does not contain the answer, the assistant is instructed to say so.
