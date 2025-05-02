from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender import SHLRecommenderRAG
from dotenv import load_dotenv
load_dotenv()
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()
recommender = SHLRecommenderRAG(groq_api_key=GROQ_API_KEY)

# Request model
class RecommendRequest(BaseModel):
    query: str

# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Recommend endpoint
@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    try:
        recommendations = recommender.recommend_with_llm(req.query)
        return {"recommended_assessments": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
