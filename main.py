from fastapi import FastAPI
from pydantic import BaseModel
from promptDeepSeek import get_answer_from_deepseek

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def query_api(data: Query):
    result = get_answer_from_deepseek(data.question)
    return {"answer": result}
