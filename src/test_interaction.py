from fastapi.testclient import TestClient
from src.server import app
from src.models import TurnInput, TaskResponse, ToolCall
from src import db
import json

# Ensure DB is initialized for the test
db.init_db()

client = TestClient(app)

def test_a2a_flow():
    print("\n--- Testing A2A Flow ---")

    # 1. Agent Discovery
    print("1. Discovery: GET /agent-card")
    response = client.get("/agent-card")
    assert response.status_code == 200
    card = response.json()
    print(f"   Green Agent: {card['name']}")
    assert "investigate_facts" in [t['name'] for t in card['tools_provided']]

    # 2. Start Task
    print("\n2. Start Task: POST /start_task")
    response = client.post("/start_task")
    if response.status_code == 500:
        print("   FAILED: No cases in DB? Run ingest first.")
        return
    assert response.status_code == 200
    task_req = response.json()
    task_id = task_req["task_id"]
    print(f"   Task ID: {task_id}")
    print(f"   Context: {task_req['initial_evidence'][:50]}...")

    # 3. Turn 1: Investigate (Tool Call)
    print("\n3. Turn 1 (Tool Call): POST /turn")
    turn_input = TurnInput(
        task_id=task_id,
        action="tool_call",
        tool_calls=[
            ToolCall(name="investigate_facts", arguments={})
        ]
    )
    # Using model_dump to serialize Pydantic model
    response = client.post("/turn", json=turn_input.model_dump())
    assert response.status_code == 200
    turn_output = response.json()
    assert turn_output["status"] == "continue"
    assert turn_output["tool_results"][0]["result"] is not None
    print(f"   Tool Result (Secret Facts): {turn_output['tool_results'][0]['result'][:50]}...")

    # 4. Turn 2: Submit
    print("\n4. Turn 2 (Submission): POST /turn")
    turn_input = TurnInput(
        task_id=task_id,
        action="submit",
        submission=TaskResponse(
            task_id=task_id,
            final_verdict="The defendant is guilty based on the evidence."
        )
    )
    response = client.post("/turn", json=turn_input.model_dump())
    assert response.status_code == 200
    turn_output = response.json()
    assert turn_output["status"] == "finished"
    print(f"   Final Score: {turn_output['score']}")
    print("--- Test Passed --- \n")

if __name__ == "__main__":
    test_a2a_flow()
