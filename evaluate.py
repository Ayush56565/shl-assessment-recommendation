import requests
import json
from tqdm import tqdm

def evaluate(benchmark_file, api_url):
    with open(benchmark_file, 'r') as f:
        benchmark_data = [json.loads(line) for line in f]

    total_recall = 0
    total_map = 0
    total_queries = len(benchmark_data)

    for data in tqdm(benchmark_data, desc="Evaluating"):
        query = data['query']
        true_assessments = set(data['urls'])  
        print(true_assessments)

        response = requests.post(f"{api_url}/recommend", json={"query": query})
        
        if response.status_code != 200:
            print(f"Error in recommendation for query: {query}")
            continue
        
        recommended_assessments = [item['url'] for item in response.json()['recommended_assessments'][:3]]
        print(recommended_assessments)
        print(true_assessments.intersection(recommended_assessments))
        recall_at_3 = len(true_assessments.intersection(recommended_assessments)) / 3
        total_recall += recall_at_3

        relevant_found = 0
        precision_at_k = 0
        for idx, rec in enumerate(recommended_assessments):
            if rec in true_assessments:
                relevant_found += 1
                precision_at_k += relevant_found / (idx + 1)
        
        map_at_3 = precision_at_k / 3
        total_map += map_at_3

    mean_recall_at_3 = total_recall / total_queries
    mean_map_at_3 = total_map / total_queries

    print(f"Mean Recall@3: {mean_recall_at_3}")
    print(f"Mean MAP@3: {mean_map_at_3}")
    
    return mean_recall_at_3, mean_map_at_3

if __name__ == "__main__":
    benchmark_file = 'eval/b.jsonl'
    api_url = 'http://localhost:8000' 
    
    evaluate(benchmark_file, api_url)
