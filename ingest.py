import os
import glob
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

DATA_DIR = "data"

def load_all_documents():
    docs = []

    # PDF files
    for pdf_path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    # Text & Markdown files
    for txt_path in glob.glob(os.path.join(DATA_DIR, "*.txt")) + \
                    glob.glob(os.path.join(DATA_DIR, "*.md")):
        loader = TextLoader(txt_path, encoding="utf-8")
        docs.extend(loader.load())

    return docs

def main():
    print("Loading documents from data/ ...")
    docs = load_all_documents()

    print(f"Loaded {len(docs)} raw documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Split into {len(chunks)} chunks.")

    client = chromadb.PersistentClient(path="chroma")
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )

    embedder = OllamaEmbeddings(model="nomic-embed-text")

    print("Embedding and storing chunks...")
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk-{i}"],
            documents=[chunk.page_content],
            embeddings=[embedder.embed_query(chunk.page_content)],
            metadatas=[{"source": chunk.metadata.get("source", "unknown")}]
        )

    print("Ingestion complete.")

if __name__ == "__main__":
    main()