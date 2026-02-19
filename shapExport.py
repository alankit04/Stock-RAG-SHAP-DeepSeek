import pickle
import shutil
import pandas as pd
import shap
import matplotlib.pyplot as plt
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

# === Generate SHAP Plot from Model ===
def generate_shap_plot():
    with open("xgb_model_subset.pkl", "rb") as f:
        model = pickle.load(f)
    X_test = pd.read_csv("X_test_subset.csv")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("Shap Plot.png", bbox_inches="tight")
    plt.close()
    print("✅ SHAP plot generated and saved to 'Shap Plot.png'")

# === Initializing ChromaDB & Embedder ===
client = chromadb.Client()
collection_name = "deepseek_rag"
collection = client.get_or_create_collection(collection_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===  Preparing the Data ===
def preprocess_and_index():
    # CSV — use feature data (X_test), not labels (y_test)
    csv_df = pd.read_csv("X_test_subset.csv")
    csv_text = "CSV Summary:\n" + csv_df.describe().to_string() + "\n\nSample Rows:\n" + csv_df.head(5).to_string()

    # Image (SHAP plot)
    image = Image.open("Shap Plot.png")
    image_text = "Image (OCR Extracted Text):\n" + pytesseract.image_to_string(image)

    # Combine
    docs = [csv_text, image_text]
    metadatas = [{"source": "csv"}, {"source": "shap_image"}]
    ids = ["doc1", "doc2"]
    embeddings = embedder.encode(docs).tolist()

    # Add to ChromaDB
    try:
        client.delete_collection(collection_name)
        # 🔁 Re-initialize the client after deletion to fix InvalidCollectionException
        new_client = chromadb.Client()
        global collection
        collection = new_client.get_or_create_collection(collection_name)
    except Exception as e:
        print("Error deleting or recreating collection:", e)

    collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)


# ===  Query + RAG + Send to DeepSeek ===
def query_rag_deepseek(user_query):
    # Embed query and retrieve context
    q_embedding = embedder.encode([user_query]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)
    context = "\n\n".join(results["documents"][0])

    # Combine context and user query
    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{user_query}
"""

    # Sending to DeepSeek via Ollama
    print("\n=== DeepSeek Response ===\n")
    subprocess.run(["ollama", "run", "deepseek-r1"], input=prompt.encode())

# === Main Loop ===
if __name__ == "__main__":
    check_ollama()
    generate_shap_plot()
    preprocess_and_index()
    print("RAGfolio is ready. How can I help you today!\n")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        query_rag_deepseek(query)
