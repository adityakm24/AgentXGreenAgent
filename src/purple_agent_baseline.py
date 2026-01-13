import requests
import json
import os
import re
from openai import OpenAI
from typing import Dict, Any, List

# Configurations
GREEN_AGENT_URL = "http://localhost:8000"
LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
# LLM Model Name
MODEL_NAME = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-4b-thinking-2507")

client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_KEY)

class PurpleAgent:
    def __init__(self):
        self.history = []
        self.task_id = None
        self.task_context = ""
    
    def connect(self):
        print("Purple Agent: Connecting...")
        try:
            resp = requests.get(f"{GREEN_AGENT_URL}/agent-card")
            card = resp.json()
            print(f"Connected to Judge: {card['name']}")
        except Exception as e:
            print(f"Connection Failed: {e}")
            exit(1)

    def start_task(self):
        print("\nPurple Agent: Requesting Case...")
        resp = requests.post(f"{GREEN_AGENT_URL}/start_task")
        task = resp.json()
        self.task_id = task['task_id']
        self.task_context = task['context']
        initial_evidence = task['initial_evidence']
        
        print(f"Case ID: {self.task_id}")
        print(f"Initial Evidence: {initial_evidence[:100]}...")
        
        # Initialize History
        self.history.append({"role": "system", "content": f"You are a Lawyer. Context: {self.task_context}"})
        self.history.append({"role": "user", "content": f"Here is the case file: {initial_evidence}. Investigate using 'Action: investigate_facts(query)' or Submit 'Final Verdict: ...'."})

    def run_loop(self):
        max_turns = 5
        for i in range(max_turns):
            print(f"\n--- Turn {i+1} ---")
            
            # 1. Think and Act
            action = self._think_and_act()
            
            if "Final Verdict:" in action:
                self._submit_verdict(action)
                break
            
            # 2. Extract Tool Call
            if "Action:" in action:
                # Parse: Action: investigate_facts("query")
                try:
                    query = action.split('investigate_facts("')[1].split('")')[0]
                except:
                    # Fallback parsing
                    query = "Give me more details."
                
                print(f"Purple Agent Decided: Ask Judge for '{query}'")
                
                # 3. Call Green Agent
                tool_result = self._call_green_agent_tool(query)
                
                # 4. Observe
                print(f"Judge Responded: {tool_result[:100]}...")
                self.history.append({"role": "user", "content": f"Observation from Judge: {tool_result}"})
            
            else:
                print("Purple Agent is confused. Retrying...")
                self.history.append({"role": "user", "content": "You must output an Action."})

    def _think_and_act(self) -> str:
        # Simple ReAct Prompting
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.history,
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": content})
        return content

    def _call_green_agent_tool(self, query: str) -> str:
        turn_input = {
            "task_id": self.task_id,
            "action": "tool_call",
            "tool_calls": [
                {
                    "name": "investigate_facts",
                    "arguments": {"query": query},
                    "id": "call_1"
                }
            ]
        }
        resp = requests.post(f"{GREEN_AGENT_URL}/turn", json=turn_input)
        data = resp.json()
        
        # Capture Judge's Thought Process message
        if data.get('message'):
            # Handle error where message might be missing or status is error
            print(f"[Judge Internal Thought]: {data.get('message', 'No thought process returned')}")
            
        if data.get('tool_results'):
            return str(data['tool_results'][0]['result'])
        return "No info."

    def _submit_verdict(self, action_text: str):
        verdict = action_text.split("Final Verdict:")[1].strip()
        print(f"\nPurple Agent Submitting Verdict: {verdict[:50]}...")
        
        resp = requests.post(f"{GREEN_AGENT_URL}/turn", json={
            "task_id": self.task_id,
            "action": "submit",
            "submission": {
                "task_id": self.task_id,
                "final_verdict": verdict
            }
        })
        result = resp.json()
        print(f"--- CASE CLOSED ---\nScore: {result['score']}")

if __name__ == "__main__":
    agent = PurpleAgent()
    agent.connect()
    agent.start_task()
    agent.run_loop()
