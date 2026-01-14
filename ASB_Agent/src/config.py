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
    
    # Evaluation Configuration  
    attack_type: str = Field(default="observation_prompt_injection", alias="ATTACK_TYPE")
    max_test_cases: int = Field(default=10, alias="MAX_TEST_CASES") # Reduced default for heavy ASB generation
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
