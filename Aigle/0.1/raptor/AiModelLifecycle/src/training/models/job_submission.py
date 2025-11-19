from pydantic import BaseModel, Field
from typing import Literal
from ..models.trainer_config import TrainerConfig, DatasetConfig


class TrainingJobSubmission(BaseModel):
    """
    Training job submission model.
    """
    task_type: Literal["instruction", "text"] = Field(
        ...,
        description="Task type: 'instruction' (Instruction Generation) or 'text' (Text Generation)"
    )

    select_multiple_gpus: bool = Field(
        ...,
        description="Whether to select multiple GPUs for training"
    )

    vram_budget_gb: float = Field(
        ...,
        gt=0,
        description="Estimated VRAM requirement (GB) for training and resource management"
    )

    training_config: TrainerConfig = Field(
        ...,
        description="Complete training configuration"
    )

    dataset_config: DatasetConfig = Field(
        ...,
        description="Complete dataset configuration"
    )

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "task_type": "instruction",
                "vram_budget_gb": 5.0,
                "select_multiple_gpus": False,
                "training_config": {
                    "model_name_or_path": "tmp/models/google_gemma-3-270m-it",
                    "use_bfloat16": True,
                    "use_flash_attn": False,
                    "weight_decay": 0.01,
                    "warmup_ratio": None,
                    "lora_config": {
                        "r": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                        "bias": "none",
                        "modules_to_save": None
                    },
                    "max_epochs": 3,
                    "batch_size": 1,
                    "learning_rate": 2e-5,
                    "gradient_accumulation_steps": 4,
                    "warmup_steps": 100,
                    "logging_steps": 5,
                    "val_check_interval": 0.5,
                    "experiment_name": "squad_finetune"
                },
                "dataset_config": {
                    "dataset_name_or_path": "rajpurkar/squad_v2",
                    "default_system_prompt": None,
                    "train_size": 100,
                    "val_ratio": None,
                    "val_size": 50,
                    "max_length": 2048,
                    "cache_dir": "tmp/datasets/rajpurkar_squad_v2"
                }
            }
        }
