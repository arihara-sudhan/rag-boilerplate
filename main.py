from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import pickle

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
g_model = genai.GenerativeModel("gemini-2.5-flash")

file_path = "corpus.txt"
corpus = Path(file_path).read_text(encoding="utf-8")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')
def get_embedding(text):
    return model.encode(text, convert_to_numpy=True).astype(np.float32)

faiss_file = "faiss_index.bin"
id_map_file = "id_to_chunk.pkl"

if os.path.exists(faiss_file) and os.path.exists(id_map_file):
    print("Loading existing FAISS index and chunks...")
    index = faiss.read_index(faiss_file)
    with open(id_map_file, "rb") as f:
        id_to_chunk = pickle.load(f)
else:
    print("Creating new FAISS index and chunks...")
    chunks = chunk_text(corpus)
    print(f"Total chunks: {len(chunks)}")
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]
    embedding_dim = chunk_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(chunk_embeddings))
    id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
    faiss.write_index(index, faiss_file)
    with open(id_map_file, "wb") as f:
        pickle.dump(id_to_chunk, f)

def retrieve(query, top_k=3):
    query_emb = get_embedding(query).astype(np.float32)
    actual_top_k = min(top_k, index.ntotal)
    D, I = index.search(np.array([query_emb]), actual_top_k)
    return "\n\n".join([id_to_chunk[i] for i in I[0] if i != -1])

def rag_query(query, top_k=3):
    #context = retrieve(query, top_k=top_k)
    prompt = f"You are stress estimator. You will assign scores in 5 scale. If the score is 5, You should ask him/her to talk with a therapist. Also alsways show the score. The user said, \n\"{query}\" You go ahead!"
    try:
        response = g_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error generating response:", e)
        return None

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or type Q to quit): ")
        if query.strip().upper() == "Q":
            break
        answer = rag_query(query)
        print("\n--- RAG Answer ---\n")
        print(answer)
