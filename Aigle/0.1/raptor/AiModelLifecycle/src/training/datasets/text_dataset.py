from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from typing import Callable
import logging

from .base_dataset import BaseDataset
from ..models.trainer_config import DatasetConfig


logger = logging.getLogger(__name__)


class TextDataset(BaseDataset):
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        config.batched_map = True
        super().__init__(config, tokenizer)

    def _get_tokenize_fn(self) -> Callable:
        def tokenize_fn(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_special_tokens_mask=True
            )
            return tokenized
        return tokenize_fn

    def _get_default_collate_fn(self) -> Callable:
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )