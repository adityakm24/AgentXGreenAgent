"""
Purple Agent Server - A2A-compatible FastAPI server for the baseline agent.

This is the agent under test that the Green Agent evaluates.
"""

import argparse
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.config import settings
from src.purple_agent.agent import PurpleAgent


# Request/Response models
class TaskRequest(BaseModel):
    """Request to start a new task."""
    user_instruction: str
    available_tools: List[Dict[str, Any]]


class ToolResultRequest(BaseModel):
    """Request with tool execution result."""
    tool_name: str
    tool_output: str
    success: bool = True


class AgentResponse(BaseModel):
    """Response from the agent."""
    status: str  # "tool_call", "complete", "error"
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    error: Optional[str] = None


# Global agent instance
agent: Optional[PurpleAgent] = None


app = FastAPI(
    title="InjecAgent Purple Agent (Baseline)",
    description="Baseline ReAct agent for InjecAgent evaluation",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    """Initialize the agent on startup."""
    global agent
    agent = PurpleAgent()
    print(f"[PurpleAgent] Server started, using model {settings.llm_model_name}")
    from src.logger import setup_logger
    logger = setup_logger("PurpleAgentServer", "purple_agent.log")
    logger.info(f"Server started on port {settings.purple_agent_port}, using model {settings.llm_model_name}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "agent": "InjecAgent Purple Agent (Baseline)"}


@app.get("/.well-known/agent.json")
async def agent_card():
    """Return the A2A agent card."""
    return {
        "name": "InjecAgent Purple Agent (Baseline)",
        "description": "A baseline ReAct agent for testing Indirect Prompt Injection vulnerabilities",
        "version": "0.1.0",
        "protocols": ["a2a"],
        "capabilities": {
            "tool_use": True,
        },
        "model": {
            "name": settings.llm_model_name,
            "provider": "LM Studio",
        }
    }


@app.post("/task", response_model=AgentResponse)
async def start_task(request: TaskRequest):
    """
    Start a new task.
    
    The Green Agent calls this to give the Purple Agent a task
    along with available tools.
    """
    global agent
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        result = agent.start_task(
            user_instruction=request.user_instruction,
            available_tools=request.available_tools,
        )
        return AgentResponse(**result)
        
    except Exception as e:
        return AgentResponse(status="error", error=str(e))


@app.post("/tool_result", response_model=AgentResponse)
async def handle_tool_result(request: ToolResultRequest):
    """
    Handle tool execution result.
    
    The Green Agent calls this after simulating a tool execution.
    Returns the agent's next action.
    """
    global agent
    
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        result = agent.handle_tool_result(
            tool_name=request.tool_name,
            tool_output=request.tool_output,
            success=request.success,
        )
        return AgentResponse(**result)
        
    except Exception as e:
        return AgentResponse(status="error", error=str(e))


@app.post("/reset")
async def reset_agent():
    """Reset the agent state."""
    global agent
    agent = PurpleAgent()
    return {"status": "ok", "message": "Agent reset"}


def main():
    """Main entry point for the Purple Agent server."""
    parser = argparse.ArgumentParser(description="InjecAgent Purple Agent Server")
    parser.add_argument("--host", default=settings.purple_agent_host)
    parser.add_argument("--port", type=int, default=settings.purple_agent_port)
    parser.add_argument("--card-url", default=None, help="URL to advertise in agent card")
    
    args = parser.parse_args()
    
    print(f"[PurpleAgent] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
