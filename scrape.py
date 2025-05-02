import requests
from bs4 import BeautifulSoup
import json
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

BASE_URL = "https://www.shl.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs("data", exist_ok=True)

def get_full_url(relative_url):
    return BASE_URL + relative_url

def extract_test_type(span_tags):
    mapping = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "S": "Simulations"
    }
    return [mapping.get(span.text.strip(), span.text.strip()) for span in span_tags]

def scrape_assessment_details(relative_url):
    full_url = get_full_url(relative_url)
    res = requests.get(full_url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.find("h1").text.strip() if soup.find("h1") else "N/A"

    description_block = soup.find("h4", string="Description")
    description = description_block.find_next_sibling("p").text.strip() if description_block else "N/A"

    duration_block = soup.find("h4", string="Assessment length")
    duration_text = duration_block.find_next_sibling("p").text if duration_block else ""
    duration = duration_text.split("=")[-1].strip().replace("minutes", "").strip() 

    remote_support_span = soup.select_one("p:contains('Remote Testing:') span.catalogue__circle")
    remote_support = "Yes" if remote_support_span and "-yes" in remote_support_span.get("class", []) else "No"

    test_type_spans = soup.select("span.product-catalogue__key[data-tooltip]")
    test_type = extract_test_type(test_type_spans)

    adaptive_support = "Yes" if "adaptive" in description.lower() or "irt" in description.lower() else "No"

    return {
        "assessment_name": title,
        "url": full_url,
        "description": description,
        "duration": duration,
        "remote_support": remote_support,
        "adaptive_support": adaptive_support,
        "test_type": test_type
    }

def scrape_and_embed():
    assessments = []
    texts_for_embedding = []
    page_count=0
    for page in range(1, 33):  
        catalog_url = f"{BASE_URL}/products/product-catalog/?start={page_count}&type=1&type=1"
        print(f"\n🔍 Scraping catalog page {page}: {catalog_url}")
        res = requests.get(catalog_url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")
    
        rows = soup.select("tr[data-entity-id]")
        print(f" → Found {len(rows)} assessments on page {page}.")

        for row in rows:
            try:
                link_tag = row.select_one("td.custom__table-heading__title a")
                if not link_tag:
                    continue
                relative_url = link_tag["href"]
                print(f"   • Scraping: {relative_url}")
                data = scrape_assessment_details(relative_url)
                assessments.append(data)

                text = (
                    f"{data['assessment_name']}. {data['description']}. "
                    f"Duration: {data['duration']} minutes. "
                    f"Test Types: {', '.join(data['test_type'])}. "
                    f"Remote Support: {data['remote_support']}. "
                    f"Adaptive Support: {data['adaptive_support']}."
                )
                texts_for_embedding.append(text)
                time.sleep(1)  
            except Exception as e:
                print(f"   ⚠️ Error scraping row: {e}")
        page_count += 12

    with open("data/scraped_data.json", "w") as f:
        json.dump(assessments, f, indent=4)
    print("Scraped data saved to data/scraped_data.json")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts_for_embedding, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "data/index.faiss")
    print("Embeddings saved to data/index.faiss")

if __name__ == "__main__":
    scrape_and_embed()
