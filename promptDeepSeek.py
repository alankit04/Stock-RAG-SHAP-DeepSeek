def get_answer_from_deepseek(question: str) -> str:
    try:
        query_embedding = embedder.encode(question)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        context = "\n".join([doc for doc in results['documents'][0]])

        prompt = f"""Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
        response = deepseek_model.generate(prompt)

        return response
    except Exception as e:
        return f"Error occurred: {str(e)}"
