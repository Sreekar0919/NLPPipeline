from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    # Claude through Bedrock (user requirement).
    # You can swap to another Bedrock-hosted Claude model id if needed.
    analyst_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    narrator_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    max_tokens: int = 1200
    temperature: float = 0.1


@dataclass
class PipelineConfig:
    output_dir: Path = Path("outputs")
    assessment_type_default: str = "SCS"
    speaker_gap_ms: int = 500
    pause_flag_ms: int = 1500
    mismatch_gate_threshold: float = 0.7
    strict_mode: bool = True
    enable_t3_research_agents: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
