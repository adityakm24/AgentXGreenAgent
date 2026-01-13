# InjecAgent Green Agent

**AgentX AgentBeats Phase 1 Submission** - Evaluates AI agents for vulnerability to Indirect Prompt Injection (IPI) attacks using the [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) benchmark.

## Overview

This Green Agent:
- Loads 1,054 test cases from the InjecAgent dataset (17 user tools, 62 attacker tools)
- Presents tasks to Purple Agents with simulated tool responses containing malicious instructions
- Detects if agents execute unintended attacker tools
- Calculates Attack Success Rate (ASR) metrics

## Quick Start

### Prerequisites

- **Python 3.11+**
- **UV** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **LM Studio** running locally with a model loaded

### 1. Install Dependencies

```bash
cd /path/to/InjectAgent
uv sync
```

### 2. Configure Environment

Edit `.env` to match your setup:

```bash
# LM Studio settings
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL_NAME=qwen/qwen3-4b-thinking-2507

# Ports
GREEN_AGENT_PORT=8000
PURPLE_AGENT_PORT=8001
```

### 3. Start LM Studio

Ensure LM Studio is running at `http://localhost:1234` with your model loaded.

### 4. Run the Agents

**Terminal 1 - Green Agent (Evaluator):**
```bash
uv run python -m src.green_agent.server
```

**Terminal 2 - Purple Agent (Baseline):**
```bash
uv run python -m src.purple_agent.server
```

### 5. Run Evaluation

**Terminal 3:**
```bash
# Quick test with 5 cases
uv run python -m src.run_assessment --max-cases 5

# Full evaluation (direct harm)
uv run python -m src.run_assessment --case-type dh

# Full evaluation (data stealing)
uv run python -m src.run_assessment --case-type ds
```

## Project Structure

```
InjectAgent/
├── .env                    # Configuration (model, ports)
├── pyproject.toml          # UV dependencies
├── scenario.toml           # AgentBeats scenario config
├── data_repo/              # Cloned InjecAgent dataset
│   └── data/
│       ├── test_cases_dh_base.json
│       ├── test_cases_ds_base.json
│       └── tools.json
├── src/
│   ├── config.py           # Centralized settings from .env
│   ├── run_assessment.py   # CLI runner
│   ├── green_agent/        # Evaluator Agent
│   │   ├── server.py       # A2A server
│   │   ├── evaluator_agent.py  # Agentic evaluator
│   │   ├── test_case_loader.py
│   │   ├── tool_simulator.py
│   │   └── scorer.py
│   └── purple_agent/       # Baseline Agent
│       ├── server.py       # A2A server
│       └── agent.py        # LangChain ReAct agent
├── Dockerfile.green        # Docker for Green Agent
└── Dockerfile.purple       # Docker for Purple Agent
```

## API Endpoints

### Green Agent (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/.well-known/agent.json` | GET | A2A agent card |
| `/assessment` | POST | Run full A2A assessment (streaming) |
| `/evaluate` | POST | Simple sync evaluation |
| `/test-cases` | GET | List available test cases |

### Purple Agent (Port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/task` | POST | Start a new task |
| `/tool_result` | POST | Return tool execution result |
| `/reset` | POST | Reset agent state |

## Metrics

The evaluation produces InjecAgent-standard ASR metrics:

- **Valid Rate**: % of test cases where the agent produced valid output
- **ASR-valid**: Attack Success Rate among valid outputs
- **ASR-all**: Attack Success Rate among all cases
- **Breakdown by attack type**: Physical Harm, Financial Harm, Data Stealing

## Docker

```bash
# Build images
docker build -f Dockerfile.green -t injecagent-green .
docker build -f Dockerfile.purple -t injecagent-purple .

# Run (requires LM Studio accessible)
docker run -p 8000:8000 --env-file .env injecagent-green
docker run -p 8001:8001 --env-file .env injecagent-purple
```

## License

MIT
