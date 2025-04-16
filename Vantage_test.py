import pandas as pd
import pytesseract
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess
import datetime
import requests
import re

# === Initialize ChromaDB & Embedder ===
client = chromadb.Client()
collection_name = "deepseek_rag"
collection = client.get_or_create_collection(collection_name)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Alpha Vantage Setup ===
API_KEY = "2JFPKWMSYBKW2JA3"
BASE_URL = "https://www.alphavantage.co/query"

# === Load Tickers from Stock_data.csv ===
def get_tickers_from_stock_data_csv(path="Stock_data.csv"):
    try:
        df = pd.read_csv(path)
        if "Ticker" in df.columns:
            tickers = df["Ticker"].dropna().astype(str).str.upper().unique().tolist()
            print(f"✅ Loaded {len(tickers)} unique tickers from {path}")
            return tickers
        else:
            print(f"❌ 'ticker' column not found in {path}")
            return []
    except Exception as e:
        print(f"❌ Error reading {path}:", e)
        return []

# === Fetch Current Stock Data & Create Doc ===
def fetch_stock_plot_and_text(symbol):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "5min",
        "apikey": API_KEY,
        "datatype": "json"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Time Series (5min)" not in data:
        raise Exception(f"No data found for {symbol}. Response: {data}")

    time_series = data["Time Series (5min)"]
    latest_time = sorted(time_series.keys())[-1]
    latest_data = time_series[latest_time]

    stock_text = (
        f"Stock Ticker: {symbol}\n"
        f"Date: {latest_time}\n"
        f"Open: {latest_data['1. open']}\n"
        f"High: {latest_data['2. high']}\n"
        f"Low: {latest_data['3. low']}\n"
        f"Close: {latest_data['4. close']}\n"
        f"Volume: {latest_data['5. volume']}"
    )

    metadata = {
        "source": "alphavantage",
        "symbol": symbol,
        "date": latest_time
    }

    return stock_text, metadata

# === Preprocess & Index ===
def preprocess_and_index():
    # CSV Summary
    csv_df = pd.read_csv("y_test_subset.csv")
    csv_text = "CSV Summary:\n" + csv_df.describe().to_string() + "\n\nSample Rows:\n" + csv_df.head(5).to_string()

    # SHAP Image Text
    image = Image.open("Shap Plot.png")
    shap_text = pytesseract.image_to_string(image)
    image_text = "Image (OCR Extracted Text):\n" + shap_text

    # Clear old data
    try:
        client.delete_collection(collection_name)
        new_client = chromadb.Client()
        global collection
        collection = new_client.get_or_create_collection(collection_name)
    except Exception as e:
        print("Error deleting or recreating collection:", e)

    # Add CSV + Image data
    base_docs = [csv_text, image_text]
    base_meta = [{"source": "csv"}, {"source": "shap_image"}]
    base_ids = ["doc1", "doc2"]
    base_embeddings = embedder.encode(base_docs).tolist()
    collection.add(documents=base_docs, metadatas=base_meta, ids=base_ids, embeddings=base_embeddings)

    # Load tickers and fetch today's data
    tickers = get_tickers_from_stock_data_csv()
    for symbol in tickers:
        try:
            stock_doc, stock_meta = fetch_stock_plot_and_text(symbol)
            collection.add(
                documents=[stock_doc],
                metadatas=[stock_meta],
                ids=[f"stock_{symbol}_{str(datetime.date.today())}"],
                embeddings=embedder.encode([stock_doc]).tolist()
            )
            print(f"✅ Added data for {symbol}")
        except Exception as e:
            print(f"⚠️ Failed to fetch or store data for {symbol}: {e}")

# === RAG + DeepSeek Call ===
def query_rag_deepseek(user_query):
    q_embedding = embedder.encode([user_query]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=5)
    context = "\n\n".join(results["documents"][0])

    prompt = f"""Use the context below to answer the question.

Context:
{context}

Question:
{user_query}
"""

    print("\n=== DeepSeek Response ===\n")
    subprocess.run(["ollama", "run", "deepseek-r1"], input=prompt.encode())

# === Main Loop ===
if __name__ == "__main__":
    preprocess_and_index()
    print("📊 RAGfolio is ready. Ask about stock insights, SHAP features, and more!\n")
    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        query_rag_deepseek(query)