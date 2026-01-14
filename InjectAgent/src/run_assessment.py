"""
Run Assessment - CLI tool to run InjecAgent evaluation.
"""

import asyncio
import argparse
import json

from src.config import settings
from src.green_agent.evaluator_agent import EvaluatorAgent
from src.green_agent.scorer import CaseResult
from src.logger import setup_logger

logger = setup_logger("AssessmentRunner", "green_agent.log")


async def run_evaluation(
    case_type: str = "dh",
    max_cases: int = 10,
    purple_agent_url: str = None,
) -> None:
    """Run the InjecAgent evaluation."""
    
    purple_url = purple_agent_url or settings.purple_agent_url
    print(f"\n{'='*60}")
    print("InjecAgent Evaluation")
    print(f"{'='*60}")
    print(f"Purple Agent URL: {purple_url}")
    print(f"Case Type: Agent Security Benchmark (ASB)")
    print(f"Max Cases: {max_cases}")
    print(f"LLM Model: {settings.llm_model_name}")
    print(f"{'='*60}\n")
    
    # Create evaluator
    evaluator = EvaluatorAgent(purple_agent_url=purple_url)
    
    # Progress callback
    async def on_progress(idx: int, result: CaseResult):
        status = "⚠️ ATTACK SUCCESS" if result.attack_successful else "✅ SAFE"
        print(f"  Case {idx + 1}: {result.case_index} - {status}")
    
    # Run evaluation
    results = await evaluator.evaluate_purple_agent(
        max_cases=max_cases,
        on_case_complete=on_progress,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    evaluator.print_results()
    
    # Save detailed results
    output_file = f"results_asb_{max_cases}.json"
    with open(output_file, "w") as f:
        json.dump(results.to_detailed_dict(), f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run ASB Evaluation")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=5,
        help="Maximum number of test cases to run"
    )
    parser.add_argument(
        "--purple-agent-url",
        default=None,
        help="URL of the Purple Agent to evaluate"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_evaluation(
        case_type="asb", # Force ASB type
        max_cases=args.max_cases,
        purple_agent_url=args.purple_agent_url,
    ))


if __name__ == "__main__":
    main()
