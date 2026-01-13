import json
import os
import re
from typing import Dict, Any, List
from openai import OpenAI
from src.models import TurnInput, TurnOutput, ToolResult
from src.prompts import INTERNAL_TOOLS_PROMPT
from src.db import get_case_by_id
from src.scoring import calculate_total_score

# Initialize LLM Client
base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
model_name = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-4b-thinking-2507")

client = OpenAI(base_url=base_url, api_key=api_key)

class JuristGreenAgent:
    def __init__(self, task_id: str, case_id: int):
        self.task_id = task_id
        self.case_id = case_id
        self.history: List[Dict[str, str]] = []
        
        # Load Case Data
        self.case_data = get_case_by_id(case_id)
        if not self.case_data:
            raise ValueError(f"Case {case_id} not found.")

    def process_turn(self, turn_input: TurnInput) -> TurnOutput:
        """
        The Core 'Agentic' Loop using ReAct.
        """
        # 1. Update History
        if turn_input.action == "tool_call" and turn_input.tool_calls:
            action_desc = f"Purple Agent requested tools: {[t.name for t in turn_input.tool_calls]} Args: {[t.arguments for t in turn_input.tool_calls]}"
        elif turn_input.action == "submit":
            action_desc = f"Purple Agent submitted verdict: {turn_input.submission.final_verdict}"
            return self._handle_submission(turn_input.submission)
        else:
            action_desc = "Purple Agent did nothing."
            
        self.history.append({"role": "user", "content": action_desc})
        
        # 2. ReAct Loop to determine response
        judge_response = self._run_react_loop(action_desc)
        
        # 3. Construct Output
        results = []
        if turn_input.tool_calls:
            for tool in turn_input.tool_calls:
                results.append(ToolResult(
                    tool_call_id=tool.id,
                    result=judge_response.get("tool_response", "No info provided."),
                    status="success" if judge_response.get("status") != "error" else "error"
                ))
        
        # Record history
        self.history.append({"role": "assistant", "content": json.dumps(judge_response)})

        return TurnOutput(
            task_id=self.task_id,
            status=judge_response.get("status", "continue"),
            tool_results=results,
            message=judge_response.get("thought_process", "The Judge is reviewing your request...")
        )

    def _run_react_loop(self, action_desc: str) -> Dict[str, Any]:
        """
        Executes the internal ReAct loop.
        Refactored to use a SINGLE message to avoid Jinja template errors in local LLMs.
        We manually append the history to the prompt string.
        """
        max_steps = 5 # Increased from 3 to give the model more room to think
        
        # Initial Prompt Construction (Manually formatted)
        prompt_text = f"""{INTERNAL_TOOLS_PROMPT}

CASE CONTEXT (Public): {self.case_data['initial_context']}
HISTORY: {str(self.history[-5:])}
CURRENT REQUEST: {action_desc}

Begin ReAct Loop:
"""
        
        for step in range(max_steps):
            try:
                # Send the entire growing prompt as a SINGLE user message
                # This bypasses the server's multi-turn template logic
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.0,
                    stop=["Observation:"] # Validation: Ensure the model stops for tool output
                )
                content = response.choices[0].message.content.strip()
                
                # Append the model's output to our manual prompt
                prompt_text += f"{content}\n"
                
                # Check if Final Answer
                if "Final Answer:" in content:
                    json_str = content.split("Final Answer:")[1].strip()
                    return self._clean_parse_json(json_str)
                
                # Check for Tool Action
                if "Action:" in content:
                    action_line = [l for l in content.split('\n') if "Action:" in l]
                    if action_line:
                        action_line = action_line[0]
                        tool_call = action_line.split("Action:")[1].strip()
                    
                        # Execute
                        observation = self._execute_internal_tool(tool_call)
                    
                        # Append Observation to manual prompt
                        prompt_text += f"Observation: {observation}\n"
                    else:
                         prompt_text += "Observation: Failed to parse Action.\n"
                
                else:
                    # Logic to force progress if model drifts
                    prompt_text += "Observation: (You must take an Action or provide a Final Answer)\n"

            except Exception as e:
                print(f"ReAct Loop Error: {e}")
                return self._error_response(str(e))
        
        return self._error_response("Max steps reached without decision.")

    def _execute_internal_tool(self, tool_call_str: str) -> str:
        """
        Parses `lookup_evidence("query")` and executes it.
        """
        try:
            if "lookup_evidence" in tool_call_str:
                # Extract query between quotes
                match = re.search(r'lookup_evidence\s*\(\s*["\'](.*?)["\']\s*\)', tool_call_str)
                if match:
                    query = match.group(1)
                    return self._lookup_evidence(query)
            elif "check_laws" in tool_call_str:
                return "Chinese Criminal Law, Civil Procedure Law."
            return "Error: Unknown tool."
        except Exception as e:
            return f"Error executing tool: {e}"

    def _lookup_evidence(self, query: str) -> str:
        """
        Semantic search (mocked with simple containment for now).
        """
        full_facts = self.case_data['full_facts']
        # Simple containment check
        # In production, use vector search here.
        if query in full_facts:
            # Return context around the match? For now just return full facts but mark as found.
            # Ideally we extract the relevant sentence.
            return f"Found in case file: {full_facts}" 
        
        # If not exact match, check for keywords
        keywords = query.split()
        hits = [word for word in keywords if word in full_facts]
        if len(hits) > 0:
             return f"Found relevant info: {full_facts}"
             
        return "Thinking: I checked the case file but found no specific mention of that."

    def _clean_parse_json(self, content: str) -> Dict[str, Any]:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
        
    def _error_response(self, error_msg: str) -> Dict[str, Any]:
         return {
            "thought_process": f"Internal Error: {error_msg}",
            "tool_response": "The Judge is unavailable.",
            "status": "error"
        }

    def _handle_submission(self, submission):
        final_score = calculate_total_score(
            submission=submission.final_verdict,
            ground_truth=self.case_data["ground_truth_reasoning"]
        )
        return TurnOutput(
            task_id=self.task_id,
            status="finished",
            score=final_score,
            message="Verdict Received. Court Adjourned."
        )
