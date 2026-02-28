import os
import glob
import chromadb
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR libraries not installed. Install with: pip install pytesseract pillow")

DATA_DIR = "data"

def load_image_with_ocr(image_path):
    """Extract text from an image using OCR."""
    if not OCR_AVAILABLE:
        return None
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return Document(page_content=text, metadata={"source": image_path})
    except Exception as e:
        print(f"⚠️  OCR failed for {image_path}: {str(e)}")
        return None

def load_all_documents(folders=None):
    docs = []
    
    # Use provided folders or default to DATA_DIR
    if folders is None:
        folders = [DATA_DIR]
    
    for folder in folders:
        # PDF files (recursive)
        for pdf_path in glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True):
            try:
                print(f"📄 Ingesting: {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️  Skipping {pdf_path}: {str(e)}")
                continue

        # Text & Markdown files (recursive)
        for txt_path in glob.glob(os.path.join(folder, "**/*.txt"), recursive=True) + \
                        glob.glob(os.path.join(folder, "**/*.md"), recursive=True):
            try:
                print(f"📄 Ingesting: {txt_path}")
                loader = TextLoader(txt_path, encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"⚠️  Skipping {txt_path}: {str(e)}")
                continue

        # Image files with OCR (recursive)
        if OCR_AVAILABLE:
            for img_path in glob.glob(os.path.join(folder, "**/*.png"), recursive=True) + \
                            glob.glob(os.path.join(folder, "**/*.jpg"), recursive=True) + \
                            glob.glob(os.path.join(folder, "**/*.jpeg"), recursive=True):
                try:
                    print(f"🖼️  Ingesting (OCR): {img_path}")
                    doc = load_image_with_ocr(img_path)
                    if doc:
                        docs.append(doc)
                except Exception as e:
                    print(f"⚠️  Skipping {img_path}: {str(e)}")
                    continue

    return docs

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--folders", "-f", nargs="+", help="Folder paths to ingest (defaults to 'data')")
    args = parser.parse_args()
    
    # Determine which folders to use
    if args.folders:
        folders = args.folders
        print(f"Ingesting from CLI-provided folders: {folders}")
    else:
        folders = [DATA_DIR]
        print(f"Using default folder: {folders}")
    
    print("Loading documents...")
    docs = load_all_documents(folders)

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