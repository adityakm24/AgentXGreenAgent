from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uuid
import logging

from src.models import (
    AgentCard, TaskRequest, TaskResponse, TurnInput, TurnOutput, ToolResult
)
from src.db import get_random_case, get_case_by_id
from src.scoring import calculate_total_score

# In-memory session store
# Maps task_id -> case_data
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Jurist-Bench Server Starting...")
    yield
    # Shutdown logic
    print("Jurist-Bench Server Shutting Down...")

app = FastAPI(lifespan=lifespan)

@app.get("/agent-card", response_model=AgentCard)
async def get_agent_card():
    return AgentCard(
        name="Jurist-Bench-Green",
        description="Evaluator for complex legal reasoning tasks (Chinese Law)",
        capabilities={
            "protocol": "A2A/1.0",
            "interaction_modes": ["synchronous"],
            "difficulty": "hard"
        },
        tools_provided=[
            {
                "name": "investigate_facts",
                "description": "Retrieves the full facts of the case (witness statements, evidence) which are not initially provided.",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
    )

@app.post("/start_task", response_model=TaskRequest)
async def start_task():
    case = get_random_case()
    if not case:
        raise HTTPException(status_code=500, detail="No cases found in database.")
    
    task_id = str(uuid.uuid4())
    sessions[task_id] = {
        "case_id": case["id"],
        "dataset_source": case["dataset_source"],
        "turn_count": 0,
        "max_turns": 5
    }
    
    return TaskRequest(
        task_id=task_id,
        context="You are a Judge in the Chinese Court system. Review the Plaintiff's request and conduct an investigation to issue a final opinion.",
        initial_evidence=case["initial_context"],
        tools=[
             {
                "name": "investigate_facts",
                "description": "Retrieves the full facts of the case (witness statements, evidence).",
                "parameters": {"type": "object", "properties": {}}
            }
        ]
    )

@app.post("/turn", response_model=TurnOutput)
async def handle_turn(turn: TurnInput):
    task_id = turn.task_id
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    session = sessions[task_id]
    case = get_case_by_id(session["case_id"])
    if turn.action == "submit":
        # Handle Submission (Scoring)
        submission = turn.submission
        case = get_case_by_id(session["case_id"])
        
        # Calculate Logic-F1 Score
        final_score = calculate_total_score(
            submission=submission.final_verdict,
            ground_truth=case["ground_truth_reasoning"]
        )
        
        return TurnOutput(
            task_id=task_id,
            status="finished",
            score=final_score,
            message=f"Case Closed."
        )
        
    elif turn.action == "tool_call":
        # Handle Tool Calls
        results = []
        if turn.tool_calls:
            for tool in turn.tool_calls:
                if tool.name == "investigate_facts":
                    # Reveal the hidden facts
                    results.append(ToolResult(
                        tool_call_id=tool.id,
                        result=case["full_facts"]
                    ))
                else:
                    results.append(ToolResult(
                        tool_call_id=tool.id,
                        result=None,
                        status="error",
                        error_message=f"Tool {tool.name} not found."
                    ))
        
        session["turn_count"] += 1
        return TurnOutput(
            task_id=task_id,
            status="continue",
            tool_results=results
        )

    raise HTTPException(status_code=400, detail="Invalid action")
