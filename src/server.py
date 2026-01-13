from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uuid
import logging

from src.models import (
    AgentCard, TaskRequest, TaskResponse, TurnInput, TurnOutput
)
from src.db import get_random_case
from src.agent_core import JuristGreenAgent

# In-memory session store
# Maps task_id -> JuristGreenAgent instance
sessions = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Jurist-Bench Server Starting (Active Judge Mode)...")
    yield
    # Shutdown logic
    print("Jurist-Bench Server Shutting Down...")

app = FastAPI(lifespan=lifespan)

@app.get("/agent-card", response_model=AgentCard)
async def get_agent_card():
    return AgentCard(
        name="Jurist-Bench-Green",
        description="Evaluator for complex legal reasoning tasks (Chinese Law). Acts as an Active Judge.",
        capabilities={
            "protocol": "A2A/1.0",
            "interaction_modes": ["synchronous"],
            "difficulty": "dynamic"
        },
        tools_provided=[
            {
                "name": "investigate_facts",
                "description": "Requests specific evidence from the Judge. Requires a 'query' argument specifying what to look for.",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What specific fact or document are you looking for?"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    )

@app.post("/start_task", response_model=TaskRequest)
async def start_task():
    case = get_random_case()
    if not case:
        raise HTTPException(status_code=500, detail="No cases found in database.")
    
    task_id = str(uuid.uuid4())
    
    # Initialize the Active Agent
    agent = JuristGreenAgent(task_id=task_id, case_id=case["id"])
    sessions[task_id] = agent
    
    return TaskRequest(
        task_id=task_id,
        context="You are a Judge/Lawyer in the Chinese Court system. Review the Plaintiff's request. WARNING: The Judge is strict. You must ask for specific evidence using the 'investigate_facts' tool with a clear 'query'.",
        initial_evidence=case["initial_context"],
        tools=[
             {
                "name": "investigate_facts",
                "description": "Requests specific evidence. argument: query (str).",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    }
                }
            }
        ]
    )

@app.post("/turn", response_model=TurnOutput)
async def handle_turn(turn: TurnInput):
    task_id = turn.task_id
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    agent = sessions[task_id]
    
    try:
        # Delegate the entire turn logic to the Agent Core
        response = agent.process_turn(turn)
        return response
    except Exception as e:
        print(f"Error processing turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))
