"""
Green Agent Server - A2A-compatible FastAPI server for the InjecAgent evaluator.

This server implements the AgentBeats A2A protocol for assessments.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import argparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config import settings
from src.green_agent.evaluator_agent import EvaluatorAgent
from src.green_agent.scorer import CaseResult


# Request/Response models
class AssessmentRequest(BaseModel):
    """A2A assessment request from the platform."""
    participants: Dict[str, str] = Field(
        description="Mapping of role names to A2A endpoint URLs"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Assessment-specific configuration"
    )


class TaskUpdate(BaseModel):
    """Task update message for streaming progress."""
    type: str  # "progress", "log", "result"
    content: Any


class AssessmentResult(BaseModel):
    """Final assessment result."""
    total_cases: int
    valid_rate: float
    asr_valid: float
    asr_all: float
    by_attack_type: Dict[str, Dict[str, Any]]
    case_results: List[Dict[str, Any]]


# Global evaluator instance
evaluator: Optional[EvaluatorAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global evaluator
    evaluator = EvaluatorAgent()
    print(f"[GreenAgent] Server started, dataset loaded from {settings.dataset_path}")
    yield
    print("[GreenAgent] Server shutting down")


app = FastAPI(
    title="InjecAgent Green Agent",
    description="Evaluates agents for Indirect Prompt Injection vulnerabilities",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "agent": "InjecAgent Green Agent"}


@app.get("/.well-known/agent.json")
async def agent_card():
    """Return the A2A agent card."""
    return {
        "name": "InjecAgent Green Agent",
        "description": "Evaluates AI agents for vulnerability to Indirect Prompt Injection (IPI) attacks using the InjecAgent benchmark",
        "version": "0.1.0",
        "protocols": ["a2a"],
        "capabilities": {
            "assessment": True,
            "streaming": True,
        },
        "evaluation": {
            "dataset": "InjecAgent",
            "metrics": ["ASR-valid", "ASR-all", "Valid Rate"],
            "attack_types": ["Direct Harm", "Data Stealing"],
        }
    }


@app.post("/assessment")
async def run_assessment(request: AssessmentRequest):
    """
    Run an InjecAgent assessment on the specified Purple Agent.
    
    This is the main A2A assessment endpoint.
    """
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    # Get Purple Agent URL from participants
    purple_agent_url = request.participants.get("purple") or request.participants.get("agent")
    if not purple_agent_url:
        raise HTTPException(
            status_code=400, 
            detail="No purple agent URL provided in participants"
        )
    
    # Update evaluator with the Purple Agent URL
    evaluator.purple_agent_url = purple_agent_url
    
    # Get config options
    config = request.config
    case_type = config.get("case_type", "dh")
    max_cases = config.get("max_cases", settings.max_test_cases)
    
    async def generate_updates():
        """Generate streaming task updates."""
        
        async def on_case_complete(idx: int, result: CaseResult):
            """Callback for each completed test case."""
            update = TaskUpdate(
                type="progress",
                content={
                    "case_index": idx,
                    "total_cases": max_cases,
                    "attack_type": result.attack_type,
                    "attack_successful": result.attack_successful,
                }
            )
            yield f"data: {json.dumps(update.model_dump())}\n\n"
        
        # Run evaluation
        try:
            results = await evaluator.evaluate_purple_agent(
                case_type=case_type,
                max_cases=max_cases,
            )
            
            # Send final result
            final_result = AssessmentResult(
                total_cases=results.total_cases,
                valid_rate=results.valid_rate,
                asr_valid=results.asr_valid,
                asr_all=results.asr_all,
                by_attack_type=results.by_attack_type,
                case_results=[r.to_dict() for r in results.case_results],
            )
            
            final_update = TaskUpdate(type="result", content=final_result.model_dump())
            yield f"data: {json.dumps(final_update.model_dump())}\n\n"
            
        except Exception as e:
            error_update = TaskUpdate(type="error", content=str(e))
            yield f"data: {json.dumps(error_update.model_dump())}\n\n"
    
    return StreamingResponse(
        generate_updates(),
        media_type="text/event-stream",
    )


@app.post("/evaluate")
async def evaluate_simple(
    purple_agent_url: Optional[str] = None,
    case_type: str = "dh",
    max_cases: int = 10,
):
    """
    Simple evaluation endpoint for testing.
    
    Unlike /assessment, this returns results synchronously.
    """
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    if purple_agent_url:
        evaluator.purple_agent_url = purple_agent_url
    
    try:
        results = await evaluator.evaluate_purple_agent(
            case_type=case_type,
            max_cases=max_cases,
        )
        
        return {
            "success": True,
            "results": results.to_dict(),
            "details": results.to_detailed_dict(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test-cases")
async def list_test_cases(
    case_type: str = "dh",
    limit: int = 10,
):
    """List available test cases for debugging."""
    global evaluator
    
    if evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")
    
    test_cases = evaluator.loader.load_test_cases(case_type=case_type, max_cases=limit)
    
    return {
        "count": len(test_cases),
        "cases": [
            {
                "index": tc.case_index,
                "attack_type": tc.attack_type,
                "user_instruction": tc.user_instruction[:100],
                "attacker_tools": tc.attacker_tools,
            }
            for tc in test_cases
        ]
    }


def main():
    """Main entry point for the Green Agent server."""
    parser = argparse.ArgumentParser(description="InjecAgent Green Agent Server")
    parser.add_argument("--host", default=settings.green_agent_host)
    parser.add_argument("--port", type=int, default=settings.green_agent_port)
    parser.add_argument("--card-url", default=None, help="URL to advertise in agent card")
    
    args = parser.parse_args()
    
    print(f"[GreenAgent] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
