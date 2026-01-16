"""
Run Assessment - CLI entry point for multi-attack ASB evaluation.

Usage:
    uv run python -m src.run_assessment --attack-method opi --max-cases 10
    uv run python -m src.run_assessment --attack-method all --max-cases 20
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.config import settings
from src.green_agent.evaluator_agent import EvaluatorAgent
from src.green_agent.vulnerability_scorer import VulnerabilityScorer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ASB (Agent Security Bench) evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Attack Methods:
  opi                 Observation Prompt Injection (default)
  dpi                 Direct Prompt Injection  
  pot_backdoor        Plan-of-Thought Backdoor Attack
  memory_attack       Memory Poisoning Attack
  all                 Run all attack types

Attack Sub-Types:
  naive               Simple direct injection (default)
  fake_completion     Pretend task is done, then inject
  escape_characters   Use escape chars to break context
  context_ignoring    Explicitly tell agent to ignore context
  combined_attack     Combination of techniques

Examples:
  # Run OPI with 5 cases
  uv run python -m src.run_assessment --attack-method opi --max-cases 5
  
  # Run all attack types with aggressive tools only
  uv run python -m src.run_assessment --attack-method all --tool-filter agg --max-cases 20
  
  # Run DPI with fake_completion injection style
  uv run python -m src.run_assessment --attack-method dpi --sub-type fake_completion
        """
    )
    
    parser.add_argument(
        "--attack-method", "-m",
        type=str,
        default=settings.attack_method,
        choices=["observation_prompt_injection", "direct_prompt_injection", 
                 "pot_backdoor", "memory_attack", "all",
                 "opi", "dpi", "mp", "pot"],  # Short aliases
        help="Attack method to use (default: observation_prompt_injection)"
    )
    
    parser.add_argument(
        "--sub-type", "-s",
        type=str,
        default=settings.attack_sub_types,
        choices=["naive", "fake_completion", "escape_characters", 
                 "context_ignoring", "combined_attack"],
        help="Attack sub-type/injection style (default: naive)"
    )
    
    parser.add_argument(
        "--tool-filter", "-t",
        type=str,
        default=settings.attack_tool_filter,
        choices=["all", "agg", "non-agg"],
        help="Filter attacker tools: all, agg (aggressive), non-agg (default: all)"
    )
    
    parser.add_argument(
        "--pot-trigger",
        type=str,
        default=None,
        help="PoT backdoor trigger phrase (default: from settings)"
    )
    
    parser.add_argument(
        "--max-cases", "-n",
        type=int,
        default=settings.max_test_cases,
        help=f"Maximum number of test cases (default: {settings.max_test_cases})"
    )
    
    parser.add_argument(
        "--purple-agent-url",
        type=str,
        default=settings.purple_agent_url,
        help=f"Purple Agent URL (default: {settings.purple_agent_url})"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results (default: results_<method>_<timestamp>.json)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def normalize_attack_method(method: str) -> str:
    """Convert short aliases to full attack method names."""
    aliases = {
        "opi": "observation_prompt_injection",
        "dpi": "direct_prompt_injection",
        "pot": "pot_backdoor",
        "mp": "memory_attack",
    }
    return aliases.get(method, method)


async def main():
    args = parse_args()
    
    # Normalize attack method
    attack_method = normalize_attack_method(args.attack_method)
    
    print("=" * 60)
    print("Agent Security Bench (ASB) Evaluation")
    print("=" * 60)
    print(f"Attack Method:  {attack_method}")
    print(f"Sub-Type:       {args.sub_type}")
    print(f"Tool Filter:    {args.tool_filter}")
    print(f"Max Cases:      {args.max_cases}")
    print(f"Purple Agent:   {args.purple_agent_url}")
    print("=" * 60)
    
    # Initialize evaluator and vulnerability scorer
    evaluator = EvaluatorAgent(purple_agent_url=args.purple_agent_url)
    vuln_scorer = VulnerabilityScorer()
    
    # Run evaluation
    start_time = datetime.now()
    
    try:
        results = await evaluator.evaluate_purple_agent(
            attack_method=attack_method,
            attack_sub_type=args.sub_type,
            tool_filter=args.tool_filter,
            pot_trigger=args.pot_trigger,
            max_cases=args.max_cases,
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Calculate vulnerability scores for each case
        for case_result in results.case_results:
            vuln_scorer.score_single_attempt(
                case_id=f"case_{case_result.case_index}",
                attack_successful=case_result.attack_successful,
                prompt_used=case_result.user_instruction,
                response=case_result.agent_response,
                tools_called=case_result.agent_tool_calls,
            )
        
        # Get vulnerability statistics
        vuln_stats = vuln_scorer.get_aggregate_stats()
        
        # Print results summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Cases:    {results.total_cases}")
        print(f"Valid Rate:     {results.valid_rate:.1f}%")
        print(f"ASR (valid):    {results.asr_valid:.1f}%")
        print(f"ASR (all):      {results.asr_all:.1f}%")
        print(f"Duration:       {duration}")
        print("=" * 60)
        
        # Print Vulnerability Score section
        print("\n" + "=" * 60)
        print("VULNERABILITY SCORE")
        print("=" * 60)
        print(f"Avg Vuln Score: {vuln_stats['average_vulnerability_score']:.2f}")
        print(f"Vulnerable:     {vuln_stats['vulnerable_count']} / {vuln_stats['total_cases']}")
        print(f"Robust:         {vuln_stats['robust_count']} / {vuln_stats['total_cases']}")
        print(f"Vuln Rate:      {vuln_stats['vulnerability_rate']:.1f}%")
        print("=" * 60)
        
        # Print breakdown by attack type
        if results.by_attack_type:
            print("\nBreakdown by Attack Type:")
            for attack_type, metrics in results.by_attack_type.items():
                print(f"  {attack_type}:")
                print(f"    - Total: {metrics.get('total', 0)}")
                print(f"    - ASR:   {metrics.get('asr', 0):.1f}%")
        
        # Save results to file
        output_file = args.output or f"results_{attack_method.split('_')[0]}_{args.max_cases}.json"
        
        output_data = {
            "config": {
                "attack_method": attack_method,
                "sub_type": args.sub_type,
                "tool_filter": args.tool_filter,
                "max_cases": args.max_cases,
            },
            "results": results.to_dict() if hasattr(results, 'to_dict') else {
                "total_cases": results.total_cases,
                "valid_rate": results.valid_rate,
                "asr_valid": results.asr_valid,
                "asr_all": results.asr_all,
                "by_attack_type": results.by_attack_type,
            },
            "vulnerability_scores": {
                "average_score": vuln_stats['average_vulnerability_score'],
                "vulnerable_count": vuln_stats['vulnerable_count'],
                "robust_count": vuln_stats['robust_count'],
                "vulnerability_rate": vuln_stats['vulnerability_rate'],
                "per_case": vuln_scorer.export_results(),
            },
            "duration_seconds": duration.total_seconds(),
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
