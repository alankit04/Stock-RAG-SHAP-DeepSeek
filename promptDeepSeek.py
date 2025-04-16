query = "What are the top features impacting response time according to the SHAP plot?"
query_embedding = embedder.encode(query)
results = collection.query(query_embeddings=[query_embedding], n_results=3)

# Extract top matching documents
context = "\n".join([doc for doc in results['documents'][0]])

# Feed into DeepSeek R1
prompt = f"""Context:\n{context}\n\nQuestion: {query}\nAnswer:"""
response = deepseek_model.generate(prompt)
print(response)
