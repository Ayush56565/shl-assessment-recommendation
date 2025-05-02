import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import re
load_dotenv()
import os

class SHLRecommenderRAG:
    def __init__(self,
                 index_path="data/index.faiss",
                 data_path="data/scraped_data.json",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 groq_api_key=None):
        self.index = faiss.read_index(index_path)
        with open(data_path, "r") as f:
            self.assessments = json.load(f)
        self.model = SentenceTransformer(embedding_model)

        if groq_api_key:
            self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        else:
            raise ValueError("GROQ API key is required")

    def retrieve_top_k(self, query, top_k=10):
        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        top_results = [self.assessments[i] for i in indices[0] if i < len(self.assessments)]
        return top_results

    def build_context(self, results):
        context = ""
        for r in results:
            context += (
                f"Assessment Name: {r['assessment_name']}\n"
                f"Description: {r['description']}\n"
                f"Duration: {r['duration']} minutes\n"
                f"Test Types: {', '.join(r['test_type'])}\n"
                f"Remote Testing Support: {r['remote_support']}\n"
                f"Adaptive/IRT Support: {r['adaptive_support']}\n"
                f"URL: {r['url']}\n"
                "----------------------\n"
            )
        return context

    def recommend_with_llm(self, query, top_k=10):
        top_results = self.retrieve_top_k(query, top_k)
        context = self.build_context(top_results)

        prompt = (
            f"You are an assessment recommendation assistant. Based on the following job description:\n\n"
            f"{query}\n\n"
            f"And the SHL assessment data below:\n\n"
            f"{context}\n\n"
            f"Recommend the top {top_k} most suitable assessments in JSON format with the following keys only:\n"
            f" - url (string)\n - adaptive_support (Yes/No)\n - description (string)\n"
            f" - duration (string)\n - remote_support (Yes/No)\n - test_type (list of strings)"
            f"Return the exact JSON data of the assessments that match as it is, don't synthesize new data."
        )
        response = self.groq.chat.completions.create(
            model="mistral-saba-24b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        raw = response.choices[0].message.content
        try:
            json_str = re.search(r"```json(.*?)```", raw, re.DOTALL)
            if json_str:
                parsed = json.loads(json_str.group(1).strip())
            else:
                parsed = json.loads(raw.strip())  
            return parsed
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON:\n" + raw)
