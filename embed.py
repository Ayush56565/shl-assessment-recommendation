# embed.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DATA_PATH = "data/scraped_data.json"
INDEX_PATH = "data/index.faiss"

def load_data():
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def prepare_texts(assessments):
    return [
        f"{a['assessment_name']}. {a['description']}. "
        f"Duration: {a['duration']} minutes. "
        f"Test Types: {', '.join(a['test_type'])}. "
        f"Remote Support: {a['remote_support']}. "
        f"Adaptive Support: {a['adaptive_support']}."
        for a in assessments
    ]

def generate_and_save_embeddings(texts):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"Embeddings saved to {INDEX_PATH}")

if __name__ == "__main__":
    assessments = load_data()
    texts = prepare_texts(assessments)
    generate_and_save_embeddings(texts)
