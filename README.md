# AgentX - Multi-Turn Green Agent 🛡️

**AgentX** is an advanced "Green Agent" designed for the [AgentBeats](https://agentbeats.dev) platform. It orchestrates the **Agent Security Benchmark (ASB)**, a rigorous test suite designed to evaluate the safety of AI agents against Indirect Prompt Injection (IPI) and other adversarial attacks.

Unlike simple single-turn evaluations, AgentX simulates complex, multi-turn interactions where attacks are embedded within realistic tool outputs (e.g., email bodies, calendar entries, database records).

---

## 🚀 Features

*   **Multi-Turn Simulation**: Evaluates agents in realistic, stateful conversations requiring multiple tool calls.
*   **Comprehensive Attack Suite**: Supports a wide range of attack vectors:
    *   **Stealthy Attacks**: Observation Prompt Injection (OPI) hidden in tool outputs.
    *   **Disruptive Attacks**: Direct Prompt Injection (DPI) and Jailbreaks.
    *   **Advanced Techniques**: Plan-of-Thought (PoT) Backdoors and Memory Poisoning.
*   **Agentic Evaluator**: Uses a sophisticated LangChain-based evaluator to analyze agent behavior, intention, and tool usage in real-time.
*   **Detailed Metrics**: Calculates Attack Success Rate (ASR) across multiple dimensions (Valid vs. All, by Attack Type) and generates a composite **Vulnerability Score (0-100)**.
*   **A2A Protocol Compliant**: Fully compatible with the Agent-to-Agent (A2A) protocol for seamless integration with AgentBeats.

---

## 📊 Leaderboard Configuration

To display your agent's results on the AgentBeats leaderboard, use the following DuckDB SQL queries:

### **1. Overall Security (Primary Metric)**
```sql
SELECT 
    id, 
    green_agent_verdict AS "Safety Verdict", 
    vulnerability_score AS "Score (0-100)", 
    summary.total_cases AS "Total Tests", 
    summary.asr_valid AS "Overall ASR (%)" 
FROM read_json_auto('results/*.json') 
ORDER BY vulnerability_score ASC
```

### **2. Attack Breakdown**
```sql
SELECT 
    id, 
    by_attack_type."Stealthy Attack".successful AS "Stealthy Hits", 
    by_attack_type."Stealthy Attack".total AS "Stealthy Total", 
    by_attack_type."Disruptive Attack".successful AS "Disruptive Hits", 
    by_attack_type."Disruptive Attack".total AS "Disruptive Total", 
    by_attack_type."Financial Harm".successful AS "Financial Hits", 
    by_attack_type."Data Stealing".successful AS "Data Stealing Hits" 
FROM read_json_auto('results/*.json') 
ORDER BY vulnerability_score ASC
```

---

## 🛠️ Architecture

AgentX consists of two main components:

1.  **Green Agent (The Evaluator)**:
    *   **Orchestrator**: Loads test cases (Synthesis) and manages the conversation flow.
    *   **Tool Simulator**: Mock environment that provides "infected" data (e.g., an email containing a prompt injection).
    *   **Scorer**: Analyzes the Purple Agent's tool calls to determine if an attack succeeded (e.g., did the agent transfer money when it shouldn't have?).

2.  **Purple Agent (The Target)**:
    *   The agent being evaluated. It connects to the Green Agent, receives instructions, and attempts to complete tasks using the provided tools.

---

## 🏃‍♂️ Running Locally

You can run the full evaluation suite locally using Docker Compose:

```bash
# 1. Clone the repository
git clone https://github.com/MenInBlack26/ASB_MultiTurn_Agents
cd ASB_MultiTurn_Agents

# 2. Build and Run
docker compose up --abort-on-container-exit
```

**Environment Variables (.env)**:
Ensure you produce a `.env` file with your LLM configuration:
```bash
LLM_BASE_URL=https://...
LLM_API_KEY=...
LLM_MODEL_NAME=gemini-2.5-flash
MAX_TEST_CASES=9999
ATTACK_METHOD=all
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
