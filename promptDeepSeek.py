import shutil
import pandas as pd
import pytesseract
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess

# === Ollama Availability Check ===
def check_ollama():
    if shutil.which("ollama") is None:
        print("❌ Ollama is not installed. Install from https://ollama.ai then run: ollama pull deepseek-r1")
        exit(1)

# === Initialize ChromaDB & Embedder ===
client = chromadb.Client()
collection_name = "deepseek_rag"
collection = client.get_or_create_collection(collection_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Prepare and Index Documents ===
def preprocess_and_index():
    # CSV — use feature data (X_test), not labels (y_test)
    csv_df = pd.read_csv("X_test_subset.csv")
    csv_text = "CSV Summary:\n" + csv_df.describe().to_string() + "\n\nSample Rows:\n" + csv_df.head(5).to_string()

    # SHAP plot text via OCR
    image = Image.open("Shap Plot.png")
    shap_text = pytesseract.image_to_string(image)
    image_text = "Image (OCR Extracted Text):\n" + shap_text

    # Clear old collection and re-create
    try:
        client.delete_collection(collection_name)
        new_client = chromadb.Client()
        global collection
        collection = new_client.get_or_create_collection(collection_name)
    except Exception as e:
        print("Error deleting or recreating collection:", e)

    docs = [csv_text, image_text]
    metadatas = [{"source": "csv"}, {"source": "shap_image"}]
    ids = ["doc1", "doc2"]
    embeddings = embedder.encode(docs).tolist()
    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)

# === Query RAG + DeepSeek via Ollama ===
def query_rag_deepseek(user_query):
    q_embedding = embedder.encode([user_query]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=3)
    context = "\n".join([doc for doc in results["documents"][0]])

    prompt = f"""Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"""

    print("\n=== DeepSeek Response ===\n")
    subprocess.run(["ollama", "run", "deepseek-r1"], input=prompt.encode())

# === Main Loop ===
if __name__ == "__main__":
    check_ollama()
    preprocess_and_index()
    print("RAGfolio is ready. Ask about SHAP features, stock predictions, and more!\n")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        query_rag_deepseek(query)
