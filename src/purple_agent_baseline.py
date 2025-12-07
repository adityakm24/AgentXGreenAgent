import requests
import json
import time

GREEN_AGENT_URL = "http://localhost:8000"

def main():
    print("Purple Agent: Connecting to Green Agent...")
    
    # 1. Discovery
    try:
        resp = requests.get(f"{GREEN_AGENT_URL}/agent-card")
        agent_card = resp.json()
        print(f"Connected to: {agent_card['name']}")
        print(f"Capabilities: {agent_card['capabilities']}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 2. Start Task
    print("\nPurple Agent: Requesting Task...")
    resp = requests.post(f"{GREEN_AGENT_URL}/start_task")
    task = resp.json()
    task_id = task['task_id']
    print(f"Received Task {task_id}")
    print(f"Context: {task['context']}")
    print(f"Initial Evidence: {task['initial_evidence'][:100]}...")

    # 3. Reasoning Loop (Simulated)
    print("\nPurple Agent: Thinking... 'I need more facts.'")
    
    # Check available tools
    tools = task['tools']
    investigate_tool = next((t for t in tools if t['name'] == 'investigate_facts'), None)
    
    if investigate_tool:
        print("Purple Agent: Using tool 'investigate_facts'...")
        turn_input = {
            "task_id": task_id,
            "action": "tool_call",
            "tool_calls": [
                {
                    "name": "investigate_facts",
                    "arguments": {},
                    "id": "call_123"
                }
            ]
        }
        resp = requests.post(f"{GREEN_AGENT_URL}/turn", json=turn_input)
        turn_output = resp.json()
        tool_results = turn_output.get('tool_results', [])
        if tool_results:
            hidden_facts = tool_results[0]['result']
            print(f"Purple Agent: Obtained new evidence: {hidden_facts[:100]}...")
        else:
            print("Purple Agent: No evidence found.")
            hidden_facts = ""
    else:
        hidden_facts = ""

    # 4. Formulate Verdict
    print("\nPurple Agent: Formulating Verdict using Qwen3...")
    
    # We use a simple prompt to simulate the agent's reasoning
    prompt = f"""You are a Judge needing to make a decision.
    
    Context: {task['context']}
    
    Initial Evidence: {task['initial_evidence']}
    
    Additional Evidence Found: {hidden_facts}
    
    Please write a Judicial Opinion (Verdict) explaining your reasoning based on the facts."""
    
    # Initialize OpenAI client for generating verdict
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        verdict = response.choices[0].message.content
    except Exception as e:
        print(f"LLM Generation Failed: {e}")
        verdict = f"Based on the investigation, the facts are: {hidden_facts[:50]}... Therefore, the defendant is liable. (Fallback due to LLM error)"
    
    # 5. Submit
    print("Purple Agent: Submitting Verdict...")
    submission_input = {
        "task_id": task_id,
        "action": "submit",
        "submission": {
            "task_id": task_id,
            "final_verdict": verdict
        }
    }
    resp = requests.post(f"{GREEN_AGENT_URL}/turn", json=submission_input)
    result = resp.json()
    
    print("\n--- Interaction Complete ---")
    print(f"Status: {result['status']}")
    print(f"Score: {result['score']}")
    print(f"Message: {result['message']}")

if __name__ == "__main__":
    main()
