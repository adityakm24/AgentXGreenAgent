import json
import re
import sys
from pathlib import Path
from src.db import init_db, insert_case

def parse_jrg_query(query_text: str):
    """
    Parses the messy string format of LAiW JRG queries.
    Format usually looks like:
    "请你... \n['诉讼请求'：(content)...\n'审理查明'：(content)...]\n"
    """
    # Remove the instructional text at the start if present
    # We mainly care about the content inside the brackets []
    
    # Try to extract content between '诉讼请求'： and '审理查明'：
    # Note: The colon might be Chinese full-width '：' or standard ':'
    
    # Extract Plaintiff's Request (Initial Context)
    req_match = re.search(r"'诉讼请求'[：:]\s*(.*?)\n\s*'审理查明'", query_text, re.DOTALL)
    if not req_match:
        # Fallback patterns or manual split if regex fails
        # Sometimes there isn't a newline
        req_match = re.search(r"'诉讼请求'[：:]\s*(.*?)'审理查明'", query_text, re.DOTALL)
    
    plaintiff_request = req_match.group(1).strip() if req_match else None

    # Extract Facts (Private Evidence)
    # Ends with ] or before the end of the logical block
    facts_match = re.search(r"'审理查明'[：:]\s*(.*?)(\]|$)", query_text, re.DOTALL)
    facts = facts_match.group(1).strip() if facts_match else None

    # Cleanup trailing quotes or brackets if regex was greedy
    if facts and facts.endswith("]"):
        facts = facts[:-1]
    
    return plaintiff_request, facts

def ingest_jrg(file_path: str):
    print(f"Ingesting JRG data from {file_path}...")
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                dataset_id = str(item.get('id', 'unknown'))
                query = item.get('query', '')
                ground_truth = item.get('answer', '')
                
                plaintiff_request, facts = parse_jrg_query(query)
                
                if not plaintiff_request or not facts:
                    print(f"Skipping ID {dataset_id}: Failed to parse query structure.")
                    continue
                
                insert_case(
                    dataset_id=dataset_id,
                    source='legal_jrg',
                    context=plaintiff_request,
                    facts=facts,
                    reasoning=ground_truth,
                    metadata={'original_query': query}
                )
                count += 1
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")
                continue
    
    print(f"Successfully ingested {count} cases from JRG.")

def main():
    init_db()
    
    jrg_path = "data/legal_jrg_test.jsonl"
    if Path(jrg_path).exists():
        ingest_jrg(jrg_path)
    else:
        print(f"File {jrg_path} not found. Run download first.")

if __name__ == "__main__":
    main()
