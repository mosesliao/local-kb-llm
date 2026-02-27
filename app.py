import chainlit as cl
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Initialize Chroma + embeddings + LLM
client = chromadb.PersistentClient(path="chroma")
collection = client.get_collection("knowledge_base")

embedder = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3")

def retrieve(query):
    query_embedding = embedder.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )
    return results["documents"][0]

@cl.on_message
async def main(message: cl.Message):
    query = message.content

    # Retrieve context
    docs = retrieve(query)
    context = "\n\n".join(docs)

    # Build prompt
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}
Answer:
"""

    # Generate response using the LLM's generate API and extract text
    result = llm.generate([prompt])
    try:
        response = result.generations[0][0].text
    except Exception:
        response = str(result)

    await cl.Message(content=response).send()