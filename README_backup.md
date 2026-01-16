# Agent Security Benchmark (ASB) - Green Agent

This repository contains the **Green Agent** (Evaluator) and a baseline **Purple Agent** for the AgentX competition, utilizing the **Agent Security Benchmark (ASB)** dataset from `agiresearch/ASB`.

The Green Agent acts as the evaluation environment, subjecting the Purple Agent to **multi-attack scenarios** including OPI, DPI, PoT Backdoor, and Memory Poisoning.

## Quick Start

```bash
# Install dependencies
cd ASB_Agent && uv sync && cd ..

# Start Green Agent (Evaluator) in one terminal
uv run --directory ASB_Agent python -m src.green_agent.server

# Start Purple Agent (target) in one terminal
uv run --directory ASB_Agent python -m src.purple_agent.server

# Run assessment in another terminal
uv run --directory ASB_Agent python -m src.run_assessment --attack-method all --max-cases 10
```

## Supported Attack Types

| Attack Method | Description |
|--------------|-------------|
| `opi` | Observation Prompt Injection - Inject into tool output |
| `dpi` | Direct Prompt Injection - Inject into user prompt |
| `pot_backdoor` | Plan-of-Thought Backdoor - Trigger-based attacks |
| `memory_attack` | Memory Poisoning - Inject into agent memory |
| `all` | Run all attack types together |

## Running Assessments

### Basic Usage

```bash
# Test Observation Prompt Injection (OPI)
uv run --directory ASB_Agent python -m src.run_assessment --attack-method opi --max-cases 5

# Test Direct Prompt Injection (DPI)
uv run --directory ASB_Agent python -m src.run_assessment --attack-method dpi --max-cases 5

# Test Plan-of-Thought Backdoor (PoT)
uv run --directory ASB_Agent python -m src.run_assessment --attack-method pot_backdoor --max-cases 5

# Test Memory Poisoning (MP)
uv run --directory ASB_Agent python -m src.run_assessment --attack-method memory_attack --max-cases 5

# Run ALL attack types together
uv run --directory ASB_Agent python -m src.run_assessment --attack-method all --max-cases 20
```

### Attack Sub-Types

Each attack method supports different injection styles:

```bash
# Use fake_completion injection style
uv run --directory ASB_Agent python -m src.run_assessment -m dpi --sub-type fake_completion

# Use escape_characters injection style
uv run --directory ASB_Agent python -m src.run_assessment -m opi --sub-type escape_characters

# Available sub-types:
# - naive (default)
# - fake_completion
# - escape_characters
# - context_ignoring
# - combined_attack
```

### Tool Filtering

Filter attacker tools by aggressiveness:

```bash
# Only aggressive attack tools
uv run --directory ASB_Agent python -m src.run_assessment -m all --tool-filter agg

# Only non-aggressive attack tools
uv run --directory ASB_Agent python -m src.run_assessment -m all --tool-filter non-agg
```

### Full Example

```bash
uv run --directory ASB_Agent python -m src.run_assessment \
  --attack-method all \
  --sub-type combined_attack \
  --tool-filter agg \
  --max-cases 50 \
  --output results_full_eval.json
```

## Project Structure

```
├── ASB_Agent/                 # Main Project Directory
│   ├── src/
│   │   ├── green_agent/       # The Evaluator Agent
│   │   │   ├── server.py
│   │   │   ├── evaluator_agent.py
│   │   │   ├── test_case_loader.py
│   │   │   ├── tool_simulator.py
│   │   │   ├── memory_store.py
│   │   │   └── scorer.py
│   │   ├── purple_agent/      # The Baseline Agent (Target)
│   │   │   ├── server.py
│   │   │   └── agent.py
│   │   ├── run_assessment.py  # CLI Entry Point
│   │   └── config.py          # Central Configuration
│   ├── data_repo/             # ASB Dataset
│   ├── Dockerfile.green       # Green Agent Container
│   ├── Dockerfile.purple      # Purple Agent Container
│   └── pyproject.toml         # Dependencies
├── Dockerfile                 # (Optional) Root Dockerfile
└── README.md                  # This file
```

## Environment Configuration

Copy `ASB_Agent/.env.example` to `ASB_Agent/.env` and configure:

```bash
# LLM Configuration
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL_NAME=qwen/qwen3-4b-thinking-2507

# Attack Configuration
ATTACK_METHOD=observation_prompt_injection
ATTACK_SUB_TYPES=naive
ATTACK_TOOL_FILTER=all

# PoT Configuration
POT_TRIGGERS=with perspicacious discernment
```

## Docker Usage

```bash
# Build and run Green Agent
docker build -t injecagent-green -f ASB_Agent/Dockerfile.green ASB_Agent
docker run -p 8000:8000 --env-file ASB_Agent/.env injecagent-green

# Build and run Purple Agent
docker build -t injecagent-purple -f ASB_Agent/Dockerfile.purple ASB_Agent
docker run -p 8001:8001 --env-file ASB_Agent/.env injecagent-purple
```

## Evaluation Metrics

- **ASR (Attack Success Rate)**: Percentage of test cases where the agent called an attacker tool
- **Valid Rate**: Percentage of test cases that completed without errors
- **By Attack Type**: Breakdown of ASR per attack method

## License

- Project: MIT
- Dataset: Refer to `agiresearch/ASB` license
