from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TrainerConfig:
    """Configuration for trainer initialization."""

    model_name_or_path: str = "google/gemma-3-270m-it"
    use_bfloat16: bool = True
    use_flash_attn: bool = False
    weight_decay: float = 0.01
    warmup_ratio: Optional[float] = None
    lora_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Basic training settings
    max_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 2e-5
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100

    # Logging and checkpointing
    logging_steps: int = 10
    val_check_interval: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # GPU settings
    accelerator: str = "auto"
    devices: int = 1
    strategy: str = "auto"
    precision: str = "16-mixed"

    # Experiment tracking
    experiment_name: str = "llm-training"
    run_name: Optional[str] = None

    # Advanced settings
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    use_cpu_offload: bool = False


@dataclass
class DatasetConfig:
    """Configuration for Dataset initialization."""
    dataset_name_or_path: str
    default_system_prompt: Optional[str] = None
    max_length: int = 2048
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    val_ratio: Optional[float] = None
    cache_dir: Optional[str] = None
    batched_map: bool = True
    seed: int = 42
