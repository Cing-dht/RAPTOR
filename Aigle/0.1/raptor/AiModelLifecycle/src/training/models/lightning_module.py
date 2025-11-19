"""
PyTorch Lightning Module wrapper for LLM training.
"""
from pathlib import Path
import torch
import lightning as L
from typing import Any, Optional, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from lightning.pytorch.utilities import rank_zero_only
import logging
import gc
import time

from ..trainers.base_trainer import TrainerConfig


logger = logging.getLogger(__name__)


class LightningLLMModule(L.LightningModule):
    """
    PyTorch Lightning Module for LLM fine-tuning with LoRA support.
    """

    def __init__(
        self, 
        config: TrainerConfig,
        cancel_token: Any = None,
        cancel_check_interval: int = 3
    ):
        super().__init__()
        self.config = config
        self.cancel_token = cancel_token
        self._cancel_check_interval = cancel_check_interval
        self._last_cancel_check_time = 0
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, # Use config field
            dtype=torch.bfloat16 if config.use_bfloat16 else torch.float16,
            use_cache=False,
            device_map="auto",
            attn_implementation="flash_attention_2" if config.use_flash_attn else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            logger.warning("Pad token not found in tokenizer. It will add a new pad token '<PAD>'.")
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('<PAD>')

        self.tokenizer.padding_side = "right"

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id 

        # Enable gradient checkpointing if specified
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA if configuration provided
        if config.lora_config:
            self._apply_lora(config.lora_config)

        # Save hyperparameters
        try:
            self.save_hyperparameters(ignore=['model', 'cancel_token'])
        except TypeError:
            from copy import copy
            self.save_hyperparameters(copy(vars(self.config)))

    def _check_for_cancellation(self):
        """Check if the cancellation token is set."""
        now = time.time()
        if now - self._last_cancel_check_time < self._cancel_check_interval:
            return

        self._last_cancel_check_time = now
        if self.cancel_token is not None and \
           hasattr(self.cancel_token, 'is_cancelled') and \
           callable(getattr(self.cancel_token, 'is_cancelled')):
            if self.cancel_token.is_cancelled():
                logger.info("Training canceled via cancel token.")
                if self.trainer is not None:
                    self.trainer.should_stop = True

                # self.teardown(stage='cancel')
                # raise KeyboardInterrupt("Training was canceled.")

    def _apply_lora(self, lora_config: Dict[str, Any]) -> None:        
        """Apply LoRA adapters to the model."""
        logger.info(f"Applying LoRA with config: {lora_config}")

        peft_config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=lora_config.get('modules_to_save', None)
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("Trainable parameters information logged above.")

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        self._check_for_cancellation()

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["input_ids"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        self._check_for_cancellation()
        
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["input_ids"].size(0))
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

        # Optional: Log learning rate
        if hasattr(self.trainer.optimizers[0], "param_groups"):
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, on_epoch=True, prog_bar=True, rank_zero_only=True)

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        epoch_loss = self.trainer.callback_metrics.get('train_loss_epoch', -1)
        logger.info(f"Epoch {self.current_epoch} completed. Train loss: {epoch_loss:.4f}")


    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and LR schedulers."""

        # Filter parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay, # Use config field
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Use estimated stepping batches (handles DDP + accumulation)
        total_steps = self.trainer.estimated_stepping_batches

        # Calculate warmup steps based on ratio or fixed value
        warmup_steps = 0
        if self.config.warmup_ratio is not None:
             warmup_steps = int(self.config.warmup_ratio * total_steps)
        else:
             warmup_steps = self.config.warmup_steps

        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self.model.eval()
        return self.model.generate(*args, **kwargs)

    @rank_zero_only
    @torch.no_grad()
    def save_hf_model(self, output_dir: str) -> None:
        """
        Merge trained LoRA weights into the base model and save the full Hugging Face model and tokenizer.
        """        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Merging LoRA weights and saving full model to {output_dir}")
        self.model.eval()

        try:
            if self.config.lora_config:
                merged_model = self.model.merge_and_unload()
            else:
                merged_model = self.model
            
            merged_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            logger.info("Full Hugging Face model (with merged LoRA weights) saved successfully.")

        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}", exc_info=True)

    def teardown(self, stage: str) -> None:
        """在訓練結束時執行資源清理"""
        logger.info(f"Teardown called for stage: {stage}. Releasing GPU memory...")

        if hasattr(self, 'trainer') and self.trainer is not None and self.trainer.optimizers:
             # 注意：這裡假設只有一個優化器
             optimizer = self.trainer.optimizers[0]
             if hasattr(optimizer, 'state'):
                  optimizer.state.clear()
                  logger.info("Optimizer state cleared.")
             optimizer.zero_grad() # 清零梯度
             del optimizer
             logger.info("Optimizer explicitly deleted.")

        # 確保模型不再位於 GPU 上
        if hasattr(self, 'model'):
            if self.model.device.type == 'cuda':
                self.model.cpu()
            del self.model
            logger.info("Model object explicitly deleted.")

        if hasattr(self, 'tokenizer'):
             del self.tokenizer
             logger.info("Tokenizer object explicitly deleted.")

        gc.collect()

        # 清除 PyTorch 的 CUDA 記憶體緩存
        if torch.cuda.is_available():
            try:
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("CUDA cache emptied.")
            except Exception as e:
                logger.warning(f"Failed to empty CUDA cache: {e}")
                
        logger.info("Garbage collection performed.")

