# Localhost based, Knowledge 📖 based LLM

## Overview

Ever tried to find that one important document buried in your digital chaos? 😵‍💫 You know, the one with the exact solution you need, hiding somewhere in your 47 different folders, 12 Google Drives, and that USB stick from 2019.

This application runs a fully local Retrieval‑Augmented Generation (RAG) LLM stack that processes documents, creates vector embeddings, and delivers precise answers.

- Ollama runs the model 🤖
- ChromaDB stores your knowledge 💾
- Chainlit provides the UI 💬
- LangChain glues everything together 🔗

## Setup

- ✅ Ensure you have Python 3.10+ installed.
- 🐙 Ensure you have Ollama installed and pull the `llama3` model:

```bash
ollama pull llama3
```

- (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

- Install dependencies (add any missing packages to `requirements.txt` first):

```bash
pip install -r requirements.txt
```

- Place all the PDF files you want to ingest into the `data` folder.

- Ingest PDFs into Chroma:

```bash
python ingest.py
```

- Run the app (example):

```bash
python app.py
```

## Future todo list:

1. Get `ingest.py` to ingest other folders in the PC.
2. Possible features to ingest data from your dropbox or other cloud storage (You can suggest😎)
