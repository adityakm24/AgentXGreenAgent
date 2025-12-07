# Jurist-Bench: Green Agent for AgentX AgentBeats

> **Abstract**: Jurist-Bench is a Green Agent (Evaluator) designed for the Berkeley RDI AgentX AgentBeats competition. It transforms the static "Legal AI in the Wild" (LAiW) dataset into a dynamic, stateful "inspection game" to evaluate the reasoning capabilities of AI agents in the Chinese legal domain. Unlike traditional benchmarks that measure simple verdict accuracy, Jurist-Bench evaluates the *process* of legal reasoning: the discovery of hidden evidence, the citation of appropriate statutes, and the logical construction of a judicial opinion. It employs a composite "Legal-F1" scoring metric ($W_1$ Fact Recall + $W_2$ Statutory Grounding + $W_3$ Logical Alignment) to reward robust, investigation-driven lawyering and penalize hallucination.

## 1. The Challenge: Complex Legal Application

Current LLM benchmarks often treat law as a text classification problem ("Guilty" vs "Innocent"). However, real-world legal practice is an **agentic workflow**:
1.  **Information Asymmetry**: Lawyers never start with all the facts. They must interview witnesses.
2.  **Tool Use**: Lawyers must search legal databases.
3.  **Reasoning**: The final verdict is useless without a sound logical argument (Syllogism).

Jurist-Bench addresses this by focusing on the **Judicial Reasoning Generation (JRG)** task from the [AC-NLG dataset](https://github.com/wuyiquan/AC-NLG), part of the LAiW benchmark.

## 2. Dataset and Architecture

### Data Source: LAiW (AC-NLG)
We utilize the Civil Procedure subset of the **Legal AI in the Wild (LAiW)** dataset.
-   **Input**: Plaintiff's Constraint (Initial Complaint).
-   **Hidden State**: Court's "Fact Finding" section (Witness testimony, evidence).
-   **Target**: The "Court's Opinion" (Reasoning).

### The Green Agent Architecture
The system follows the **Agent-to-Agent (A2A) Protocol**:

1.  **Handshake**: The Purple Agent connects to the Green Agent server.
2.  **Discovery (Turn 0)**: The Green Agent presents the *Initial Complaint* but hides the *Fact Finding* section.
3.  **Investigation (Turn 1)**: The Purple Agent must realize information is missing and call the `investigate_facts` tool.
4.  **Submission (Turn 2)**: The Purple Agent submits a final Verdict.
5.  **Judgment**: The Green Agent scores the submission using the **Legal-F1** metric.

### Scoring Engine: Legal-F1
The score is a weighted average of three components:
$$ Score = 0.3 \times \text{Recall} + 0.3 \times \text{Grounding} + 0.4 \times \text{Alignment} $$

-   **Fact Recall**: Did the agent mention key entities (dates, amounts) found in the hidden evidence? (Prevents Hallucination)
-   **Statutory Grounding**: Did the agent cite the relevant Law Articles (e.g., "Article 24")? (Ensures Validity)
-   **Logical Alignment**: An **AI Judge** (powered by Qwen/GPT) compares the semantic reasoning of the agent's verdict against the ground truth. (Ensures Logic)

## 3. How to Work with This Code

### 1. Setup Environment
First, ensure you have Python 3.10+ installed. Activate your virtual environment and install dependencies:

```bash
# Activate Virtual Environment (Example)
source .venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
# OR
uv sync
```

### 2. Configure Settings
Create a `.env` file to point to your LLM (using simple local hosting software like LM Studio):
```bash
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=openai/gpt-oss-20b
PORT=8000
```

### 3. Initialize Data
If running for the first time, you need to populate the database:
```bash
# Download data
curl -L "https://huggingface.co/datasets/daishen/legal-jrg/resolve/main/test.jsonl" -o data/legal_jrg_test.jsonl

# Ingest into SQLite
python -m src.ingest
```

### 4. Run the Green Agent (Server)
Start the evaluation server in one terminal:
```bash
python main.py
```
*You will see: "Starting Green Agent Server on port 8000..."*

### 5. Run the Purple Agent (Competitor)
In a **separate** terminal, run the baseline agent script. This script connects to the server and attempts to solve a legal case.

```bash
python -m src.purple_agent_baseline
```

### Example Output
Here is what a successful interaction looks like:

```text
(greenagent) adityakm@Adityas-MacBook-Pro GreenAgent % python -m src.purple_agent_baseline 
Purple Agent: Connecting to Green Agent...
Connected to: Jurist-Bench-Green
Capabilities: {'protocol': 'A2A/1.0', 'interaction_modes': ['synchronous'], 'difficulty': 'hard'}

Purple Agent: Requesting Task...
Received Task fd1e03f3-0a80-44b4-bda0-3f7f720f02b4
Context: You are a Judge in the Chinese Court system. Review the Plaintiff's request and conduct an investigation to issue a final opinion.
Initial Evidence: 为被告立即偿还借款16000元并赔偿利息损失（自起诉之日起按银行同期同类贷款利息计算至判决确定履行之日止）。。2、被告承担本案诉讼费...

Purple Agent: Thinking... 'I need more facts.'
Purple Agent: Using tool 'investigate_facts'...
Purple Agent: Obtained new evidence: 经审理，本院认定事实如下：原、被告系朋友关系。2011年期间，被告向原告借款2万元，后被告偿还部分款项。2011年9月26日，经原、被告结算，被告向原告出具借款16000元的借条一张，未约定利息及借款...

Purple Agent: Formulating Verdict using Qwen3...
Purple Agent: Submitting Verdict...

--- Interaction Complete ---
Status: finished
Score: {'total': 0.6, 'recall': 1.0, 'grounding': 1.0, 'alignment': 0.0}
Message: Case Closed.
```

## 4. Directory Structure

```
GreenAgent/
├── main.py                 # Entry point for the Green Agent Server
├── Dockerfile              # Container definition
├── src/
│   ├── server.py           # FastAPI server (A2A Protocol)
│   ├── scoring.py          # Legal-F1 Scoring Logic (Fact Recall + AI Judge)
│   ├── db.py               # SQLite Database Handler
│   ├── ingest.py           # LAiW Data Ingestion Script
│   ├── models.py           # Pydantic Models for A2A
│   └── purple_agent_baseline.py # Reference Competitor Agent
└── data/
    └── jurist.db           # SQLite Database (generated)
```

## License
MIT License.
Data derived from [LAiW](https://github.com/Dai-shen/LAiW) (Apache-2.0).