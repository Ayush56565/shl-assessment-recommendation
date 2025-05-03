from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender import SHLRecommenderRAG
from dotenv import load_dotenv
load_dotenv()
import os
import uvicorn

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()
recommender = SHLRecommenderRAG(groq_api_key=GROQ_API_KEY)

class RecommendRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Welcome to the SHL Assessment Recommender API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    try:
        recommendations = recommender.recommend_with_llm(req.query)
        return {"recommended_assessments": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)