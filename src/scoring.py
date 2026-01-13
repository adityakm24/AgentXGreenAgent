import re
import os
import math
from typing import Dict, Set, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI Client for Local LLM
base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
model_name = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-4b-thinking-2507")

try:
    client = OpenAI(base_url=base_url, api_key=api_key)
except Exception:
    client = None
    print("Warning: Could not initialize OpenAI client. Logical Alignment score will be 0.")

def extract_entities(text: str) -> Set[str]:
    """Extracts dates, money amounts, and citation numbers as 'facts'."""
    entities = set()
    
    # Dates: 2012年9月7日
    dates = re.findall(r"\d{4}年\d{1,2}月\d{1,2}日", text)
    entities.update(dates)
    
    # Money: 20万元, 50000元
    money = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?(?:万)?元", text)
    entities.update(money)
    
    return entities

def extract_citations(text: str) -> Set[str]:
    """Extracts Article numbers e.g. 第X条."""
    # Matches 《...》 or just Article X
    # Focusing on Article numbers: 第[0-9]+条
    citations = re.findall(r"第\d+条", text)
    return set(citations)

def calculate_fact_recall(submission: str, ground_truth: str) -> float:
    gt_entities = extract_entities(ground_truth)
    if not gt_entities:
        return 1.0 # No specifics to recall
    
    sub_entities = extract_entities(submission)
    intersection = sub_entities.intersection(gt_entities)
    
    return len(intersection) / len(gt_entities)

def calculate_grounding(submission: str, ground_truth: str) -> float:
    gt_citations = extract_citations(ground_truth)
    if not gt_citations:
        # If GT has no citations, did the agent hallucinate any?
        # If agent cites nothing, score 1. If agent cites something, score 0.5 (maybe irrelevant).
        return 1.0
    
    sub_citations = extract_citations(submission)
    intersection = sub_citations.intersection(gt_citations)
    
    # Jaccard Index for citations
    union = sub_citations.union(gt_citations)
    if not union:
        return 0.0
        
    return len(intersection) / len(union)

def calculate_logical_alignment(submission: str, ground_truth: str) -> float:
    if not client:
        return 0.0
        
    prompt = f"""Compare the following two legal arguments. output a single float score from 0.0 to 1.0 indicating how semantically similar the reasoning is.
    
    Ground Truth: "{ground_truth}"
    
    Submission: "{submission}"
    
    Score (0.0 to 1.0):"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        # Extract float
        match = re.search(r"0\.\d+|1\.0|0|1", content)
        if match:
            return float(match.group(0))
        return 0.5 # Fallback
    except Exception as e:
        print(f"LLM Error: {e}")
        return 0.0

def calculate_total_score(submission: str, ground_truth: str) -> Dict[str, float]:
    w1, w2, w3 = 0.3, 0.3, 0.4
    
    recall = calculate_fact_recall(submission, ground_truth)
    grounding = calculate_grounding(submission, ground_truth)
    alignment = calculate_logical_alignment(submission, ground_truth)
    # MOCK ALIGNMENT FOR NOW TO AVOID NETWORK CALLS/LATENCY DURING DEV
    # alignment = 0.8
    
    total = (w1 * recall) + (w2 * grounding) + (w3 * alignment)
    
    return {
        "total": round(total, 4),
        "recall": round(recall, 4),
        "grounding": round(grounding, 4),
        "alignment": round(alignment, 4)
    }
