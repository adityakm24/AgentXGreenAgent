import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.config import settings

@dataclass
class TestCase:
    """Represents a synthesized ASB test case."""
    case_id: str
    user_task: str
    agent_name: str
    
    # Attacker Information
    attacker_tool_name: str
    attacker_instruction: str
    attack_goal: str
    attack_type: str
    
    # Tool Definitions
    normal_tools: List[Dict[str, Any]]
    attacker_tools: List[Dict[str, Any]] # Usually just the one attacker tool, but list for consistency

    def __post_init__(self):
        # Ensure tools are in the format expected by LangChain/LLMs if needed, 
        # or we render them later. For now, we store raw dicts.
        pass

class TestCaseLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.dataset_dir
        self.user_tasks_path = self.data_dir / settings.asb_user_tasks_file
        self.attacker_tools_path = self.data_dir / settings.asb_attacker_tools_file
        self.normal_tools_path = self.data_dir / settings.asb_normal_tools_file
        
    def load_user_tasks(self) -> List[Dict[str, Any]]:
        """Load user tasks (agent_task.jsonl)."""
        tasks = []
        with jsonlines.open(self.user_tasks_path) as reader:
            for obj in reader:
                tasks.append(obj)
        return tasks

    def load_attacker_tools(self) -> List[Dict[str, Any]]:
        """Load attacker tools (all_attack_tools.jsonl)."""
        tools = []
        with jsonlines.open(self.attacker_tools_path) as reader:
            for obj in reader:
                tools.append(obj)
        return tools

    def load_normal_tools(self) -> List[Dict[str, Any]]:
        """Load normal tools (all_normal_tools.jsonl)."""
        tools = []
        with jsonlines.open(self.normal_tools_path) as reader:
            for obj in reader:
                tools.append(obj)
        return tools

    def get_normal_tools_for_agent(self, agent_name: str, all_normal_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter normal tools by agent name."""
        return [t for t in all_normal_tools if t.get("Corresponding Agent") == agent_name]

    def get_attacker_tools_for_agent(self, agent_name: str, all_attacker_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter attacker tools by agent name."""
        return [t for t in all_attacker_tools if t.get("Corresponding Agent") == agent_name]

    def load_test_cases(self, agent_filter: Optional[str] = None, limit: int = 100) -> List[TestCase]:
        """
        Synthesize test cases by crossing User Tasks with Attacker Tools.
        Logic:
           ForEach Agent in agent_task.jsonl:
             ForEach Task in Agent.Tasks:
               ForEach AttackerTool in Agent.AttackerTools:
                  Create TestCase(Task + AttackerTool)
        """
        user_agents = self.load_user_tasks()
        all_attacker_tools = self.load_attacker_tools()
        all_normal_tools = self.load_normal_tools()
        
        test_cases = []
        case_counter = 0

        for agent_obj in user_agents:
            agent_name = agent_obj.get("agent_name")
            
            # Filter by agent if requested
            if agent_filter and agent_name != agent_filter:
                continue
                
            tasks = agent_obj.get("tasks", [])
            
            # Get relevant tools
            agent_attacker_tools = self.get_attacker_tools_for_agent(agent_name, all_attacker_tools)
            agent_normal_tools = self.get_normal_tools_for_agent(agent_name, all_normal_tools)
            
            if not agent_attacker_tools:
                print(f"[Warning] No attacker tools found for agent: {agent_name}")
                continue

            for task_str in tasks:
                # Create a test case for every possible attacker tool
                for atk_tool in agent_attacker_tools:
                    if case_counter >= limit:
                        return test_cases
                        
                    tc = TestCase(
                        case_id=f"{agent_name}_{case_counter}",
                        user_task=task_str,
                        agent_name=agent_name,
                        attacker_tool_name=atk_tool.get("Attacker Tool"),
                        attacker_instruction=atk_tool.get("Attacker Instruction"),
                        attack_goal=atk_tool.get("Attack goal"),
                        attack_type=atk_tool.get("Attack Type"),
                        normal_tools=agent_normal_tools,
                        attacker_tools=[atk_tool] # We expose the attacker tool so the agent *can* call it
                    )
                    test_cases.append(tc)
                    case_counter += 1
                    
        return test_cases
