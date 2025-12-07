from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import uuid

# --- A2A Protocol Basics ---

class AgentCard(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    capabilities: Dict[str, Any]
    tools_provided: List[Dict[str, Any]] = []

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class ToolResult(BaseModel):
    tool_call_id: str
    result: Any
    status: Literal["success", "error"] = "success"
    error_message: Optional[str] = None

# --- Jurist-Bench Specific ---

class TaskRequest(BaseModel):
    """The initial "Challenge" sent to the Purple Agent."""
    task_id: str
    context: str = "You are a Judge/Lawyer in the Chinese Court system..."
    initial_evidence: str
    tools: List[Dict[str, Any]] # Function definitions

class TaskResponse(BaseModel):
    """The Purple Agent's final submission."""
    task_id: str
    reasoning_trace: Optional[str] = None
    final_verdict: str  # The generated "Court Opinion"

class TurnInput(BaseModel):
    """Input for a single turn (Purple Agent -> Green Agent)."""
    task_id: str
    action: Literal["tool_call", "submit"]
    tool_calls: Optional[List[ToolCall]] = None
    submission: Optional[TaskResponse] = None

class TurnOutput(BaseModel):
    """Output for a single turn (Green Agent -> Purple Agent)."""
    task_id: str
    status: Literal["continue", "finished", "error"]
    tool_results: Optional[List[ToolResult]] = None
    score: Optional[Dict[str, float]] = None # Only returned on 'finished'
    message: Optional[str] = None
