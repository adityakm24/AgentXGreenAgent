import sqlite3
import json
from typing import List, Dict, Any, Optional

DB_PATH = "data/jurist.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the SQLite database with the required schema."""
    conn = get_connection()
    c = conn.cursor()
    
    # Cases table
    # initial_context: The text presented to the agent at the start (e.g., Plaintiff's complaint)
    # full_facts: The text containing the evidence (e.g., Court's finding of facts)
    # ground_truth_reasoning: The target reasoning (e.g., Court's opinion)
    c.execute('''
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT,
            dataset_source TEXT,
            initial_context TEXT, 
            full_facts TEXT,
            ground_truth_reasoning TEXT,
            metadata TEXT
        )
    ''')

    # Evidence table (Optional: if we later perform finer-grained extraction)
    c.execute('''
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER,
            evidence_type TEXT,
            content TEXT,
            FOREIGN KEY(case_id) REFERENCES cases(id)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def insert_case(dataset_id: str, source: str, context: str, facts: str, reasoning: str, metadata: Dict[str, Any] = None):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO cases (dataset_id, dataset_source, initial_context, full_facts, ground_truth_reasoning, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (dataset_id, source, context, facts, reasoning, json.dumps(metadata or {})))
    conn.commit()
    conn.close()

def get_case_by_id(case_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cases WHERE id = ?', (case_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_random_case() -> Optional[Dict[str, Any]]:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM cases ORDER BY RANDOM() LIMIT 1')
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None
