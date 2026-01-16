"""
Test Case Loader - Multi-Attack Scenario Support

Loads test cases from the ASB dataset with support for:
- OPI (Observation Prompt Injection)
- DPI (Direct Prompt Injection)
- PoT (Plan-of-Thought Backdoor)
- MP (Memory Poisoning)
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
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
    attack_type: str  # "Stealthy Attack" or "Disruptive Attack"
    
    # Attack Method (NEW)
    attack_method: str = "observation_prompt_injection"  # dpi, opi, pot_backdoor, memory_attack
    attack_sub_type: str = "naive"  # fake_completion, escape_characters, context_ignoring, combined_attack
    
    # PoT Backdoor specific (NEW)
    pot_trigger: Optional[str] = None
    pot_messages: Optional[List[str]] = None  # Reasoning step messages
    
    # Tool Definitions
    normal_tools: List[Dict[str, Any]] = field(default_factory=list)
    attacker_tools: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggressive flag
    is_aggressive: bool = False


class TestCaseLoader:
    """
    Loads and synthesizes test cases for multi-attack evaluation.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.dataset_dir
        self.user_tasks_path = self.data_dir / settings.asb_user_tasks_file
        self.attacker_tools_path = self.data_dir / settings.asb_attacker_tools_file
        self.normal_tools_path = self.data_dir / settings.asb_normal_tools_file
        self.pot_tasks_path = self.data_dir / settings.pot_task_file
        self.pot_msg_path = self.data_dir / settings.pot_msg_file
        
    def load_user_tasks(self, use_pot: bool = False) -> List[Dict[str, Any]]:
        """Load user tasks from agent_task.jsonl or agent_task_pot.jsonl."""
        path = self.pot_tasks_path if use_pot else self.user_tasks_path
        tasks = []
        try:
            with jsonlines.open(path) as reader:
                for obj in reader:
                    tasks.append(obj)
        except Exception as e:
            print(f"[Warning] Failed to load tasks from {path}: {e}")
        return tasks

    def load_attacker_tools(self, tool_filter: str = "all") -> List[Dict[str, Any]]:
        """
        Load attacker tools with optional filtering.
        
        Args:
            tool_filter: "all", "agg" (aggressive only), "non-agg" (non-aggressive only)
        """
        tools = []
        with jsonlines.open(self.attacker_tools_path) as reader:
            for obj in reader:
                if tool_filter == "all":
                    tools.append(obj)
                elif tool_filter == "agg" and obj.get("Aggressive") == "True":
                    tools.append(obj)
                elif tool_filter == "non-agg" and obj.get("Aggressive") == "False":
                    tools.append(obj)
        return tools

    def load_normal_tools(self) -> List[Dict[str, Any]]:
        """Load normal tools (all_normal_tools.jsonl)."""
        tools = []
        with jsonlines.open(self.normal_tools_path) as reader:
            for obj in reader:
                tools.append(obj)
        return tools
    
    def load_pot_messages(self) -> Dict[str, Dict[str, str]]:
        """Load PoT reasoning step messages."""
        messages = {}
        try:
            with jsonlines.open(self.pot_msg_path) as reader:
                for obj in reader:
                    agent_name = obj.get("agent_name")
                    messages[agent_name] = obj
        except Exception as e:
            print(f"[Warning] Failed to load PoT messages: {e}")
        return messages

    def get_normal_tools_for_agent(self, agent_name: str, all_normal_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter normal tools by agent name."""
        return [t for t in all_normal_tools if t.get("Corresponding Agent") == agent_name]

    def get_attacker_tools_for_agent(self, agent_name: str, all_attacker_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter attacker tools by agent name."""
        return [t for t in all_attacker_tools if t.get("Corresponding Agent") == agent_name]

    def load_test_cases(
        self, 
        attack_method: str = "observation_prompt_injection",
        attack_sub_type: str = "naive",
        tool_filter: str = "all",
        pot_trigger: Optional[str] = None,
        agent_filter: Optional[str] = None, 
        limit: int = 100
    ) -> List[TestCase]:
        """
        Synthesize test cases with attack method awareness.
        
        Args:
            attack_method: dpi, opi, pot_backdoor, memory_attack, or "all"
            attack_sub_type: naive, fake_completion, escape_characters, context_ignoring, combined_attack
            tool_filter: all, agg, non-agg
            pot_trigger: Backdoor trigger phrase for PoT attacks
            agent_filter: Filter by specific agent name
            limit: Maximum number of test cases
        """
        # Handle "all" attack methods
        if attack_method == "all":
            all_cases = []
            methods = ["observation_prompt_injection", "direct_prompt_injection", "pot_backdoor", "memory_attack"]
            # Ensure at least 1 case per method when limit is small
            per_method_limit = max(1, limit // len(methods))
            
            for method in methods:
                cases = self.load_test_cases(
                    attack_method=method,
                    attack_sub_type=attack_sub_type,
                    tool_filter=tool_filter,
                    pot_trigger=pot_trigger,
                    agent_filter=agent_filter,
                    limit=per_method_limit
                )
                all_cases.extend(cases)
            return all_cases[:limit]
        
        # PoT Backdoor uses different task file
        use_pot = attack_method == "pot_backdoor"
        user_agents = self.load_user_tasks(use_pot=use_pot)
        all_attacker_tools = self.load_attacker_tools(tool_filter=tool_filter)
        all_normal_tools = self.load_normal_tools()
        pot_messages = self.load_pot_messages() if use_pot else {}
        
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
            
            # Get PoT messages for this agent if applicable
            agent_pot_msgs = pot_messages.get(agent_name, {})
            pot_msg_list = [
                agent_pot_msgs.get("msg1", ""),
                agent_pot_msgs.get("msg2", ""),
                agent_pot_msgs.get("msg3", ""),
                agent_pot_msgs.get("msg4", ""),
            ] if agent_pot_msgs else None

            for task_str in tasks:
                # Create a test case for every possible attacker tool
                for atk_tool in agent_attacker_tools:
                    if case_counter >= limit:
                        return test_cases
                        
                    tc = TestCase(
                        case_id=f"{agent_name}_{attack_method}_{case_counter}",
                        user_task=task_str,
                        agent_name=agent_name,
                        attacker_tool_name=atk_tool.get("Attacker Tool"),
                        attacker_instruction=atk_tool.get("Attacker Instruction"),
                        attack_goal=atk_tool.get("Attack goal"),
                        attack_type=atk_tool.get("Attack Type"),
                        attack_method=attack_method,
                        attack_sub_type=attack_sub_type,
                        pot_trigger=pot_trigger,
                        pot_messages=pot_msg_list,
                        normal_tools=agent_normal_tools,
                        attacker_tools=[atk_tool],
                        is_aggressive=atk_tool.get("Aggressive") == "True",
                    )
                    test_cases.append(tc)
                    case_counter += 1
                    
        return test_cases
