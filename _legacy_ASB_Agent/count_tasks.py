from src.green_agent.test_case_loader import TestCaseLoader
import os

os.environ["DATASET_PATH"] = "/Users/adityakm/Documents/Projects/AgentXGreenAgent/ASB_Agent/data_repo/data"

loader = TestCaseLoader()
test_cases = loader.load_test_cases(limit=10000) # Large limit to get everything
print(f"Total synthesized test cases: {len(test_cases)}")

user_agents = loader.load_user_tasks()
print(f"Total agents: {len(user_agents)}")
total_tasks = sum(len(a.get('tasks', [])) for a in user_agents)
print(f"Total unique user tasks: {total_tasks}")

all_atk_tools = loader.load_attacker_tools()
print(f"Total attacker tools: {len(all_atk_tools)}")
