"""
InjecAgent Green Agent - Configuration Module

Centralized configuration loading from .env file.
All settings can be changed in .env and will apply across the codebase.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    llm_base_url: str = Field(default="http://localhost:1234/v1", alias="LLM_BASE_URL")
    llm_model_name: str = Field(default="qwen/qwen3-4b-thinking-2507", alias="LLM_MODEL_NAME")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    
    # Agent Server Configuration
    green_agent_host: str = Field(default="0.0.0.0", alias="GREEN_AGENT_HOST")
    green_agent_port: int = Field(default=8000, alias="GREEN_AGENT_PORT")
    purple_agent_host: str = Field(default="0.0.0.0", alias="PURPLE_AGENT_HOST")
    purple_agent_port: int = Field(default=8001, alias="PURPLE_AGENT_PORT")
    
    # Dataset Configuration (ASB)
    dataset_path: str = Field(default="./data_repo/data", alias="DATASET_PATH")
    asb_user_tasks_file: str = Field(default="agent_task.jsonl", alias="ASB_USER_TASKS_FILE")
    asb_attacker_tools_file: str = Field(default="all_attack_tools.jsonl", alias="ASB_ATTACKER_TOOLS_FILE")
    asb_normal_tools_file: str = Field(default="all_normal_tools.jsonl", alias="ASB_NORMAL_TOOLS_FILE")
    
    # Attack Configuration
    attack_method: str = Field(
        default="observation_prompt_injection", 
        alias="ATTACK_METHOD",
        description="Attack method: observation_prompt_injection, direct_prompt_injection, pot_backdoor, memory_attack, all"
    )
    attack_sub_types: str = Field(
        default="naive",
        alias="ATTACK_SUB_TYPES",
        description="Comma-separated sub-types: naive, fake_completion, escape_characters, context_ignoring, combined_attack"
    )
    attack_tool_filter: str = Field(
        default="all",
        alias="ATTACK_TOOL_FILTER",
        description="Filter attacker tools: all, agg (aggressive only), non-agg (non-aggressive only)"
    )
    
    # PoT Backdoor Configuration
    pot_triggers: str = Field(
        default="with perspicacious discernment",
        alias="POT_TRIGGERS",
        description="Comma-separated backdoor trigger phrases"
    )
    pot_task_file: str = Field(default="agent_task_pot.jsonl", alias="POT_TASK_FILE")
    pot_msg_file: str = Field(default="agent_task_pot_msg.jsonl", alias="POT_MSG_FILE")
    
    # Evaluation Configuration  
    max_test_cases: int = Field(default=10, alias="MAX_TEST_CASES")
    max_agent_iterations: int = Field(default=15, alias="MAX_AGENT_ITERATIONS")
    
    # Logging
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    @property
    def dataset_dir(self) -> Path:
        """Get the dataset directory as a Path object."""
        return Path(self.dataset_path)
    
    @property
    def green_agent_url(self) -> str:
        """Get the full Green Agent URL."""
        return f"http://{self.green_agent_host}:{self.green_agent_port}"
    
    @property
    def purple_agent_url(self) -> str:
        """Get the full Purple Agent URL."""
        return f"http://{self.purple_agent_host}:{self.purple_agent_port}"


# Global settings instance
settings = Settings()
