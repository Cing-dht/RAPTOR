from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable, Callable
from datasets import DatasetDict
import logging
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset
from pathlib import Path

from ..models.trainer_config import DatasetConfig


logger = logging.getLogger(__name__)


@runtime_checkable
class DataLoaderProtocol(Protocol):
    def __iter__(self): ...
    def __len__(self) -> int: ...


class BaseDataset(ABC):
    """
    Production-grade abstract base class for all dataset loaders.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        **kwargs
    ):
        self.config = config
        self.tokenizer = tokenizer

        self.dataset: Optional[DatasetDict] = None
        self.raw_dataset: Optional[DatasetDict] = None
        self._tokenized = False

    @abstractmethod
    def _get_tokenize_fn(self) -> Callable:
        """
        Subclasses must implement this to return the specific tokenization/labeling function.
        """
        pass
    
    @abstractmethod
    def _get_default_collate_fn(self) -> Callable:
        """
        Subclasses must implement this to return their task-specific default DataCollator.
        """
        pass

    def _load_raw_data(self) -> DatasetDict:
        data_path = self.config.dataset_name_or_path
        if Path(data_path).exists():
            if data_path.endswith((".json", ".jsonl")):
                raw_data = load_dataset("json", data_files=data_path, split="train", cache_dir=self.config.cache_dir)
            else:
                raw_data = load_dataset("text", data_files=data_path, split="train", cache_dir=self.config.cache_dir)

            return DatasetDict({"train": raw_data})
        
        else:
            logger.info(f"Loading raw text dataset {data_path} from huggingface hub")
            dataset_dict = load_dataset(data_path, cache_dir=self.config.cache_dir)

            if not isinstance(dataset_dict, DatasetDict):
                raise TypeError(
                    f"Expected DatasetDict from {data_path}, but got {type(dataset_dict)}. "
                    f"Ensure you don't use the 'split' argument when loading the main dataset dict."
                )
            
            if "train" not in dataset_dict:
                available_splits = list(dataset_dict.keys())
                
                if len(available_splits) == 1:
                    sole_split_name = available_splits[0]
                    logger.warning(
                        f"Dataset {data_path} does not have a 'train' split. "
                        f"Renaming sole split '{sole_split_name}' to 'train' for processing."
                    )
                    dataset_dict["train"] = dataset_dict.pop(sole_split_name)
                else:
                    raise ValueError(
                        f"HuggingFace dataset {data_path} must contain a 'train' split. "
                        f"Available splits are: {available_splits}. Cannot proceed."
                    )
                        
            if "validation" not in dataset_dict and "test" in dataset_dict:
                dataset_dict["validation"] = dataset_dict.pop("test")
            
            return dataset_dict

    def _split_and_limit_dataset(self, raw_dataset: DatasetDict) -> DatasetDict:
        """
        Split the validation set and limit the training set size.
        - Validation set split: only split 'train' into 'train' and 'validation' sets (Priority: val_ratio > val_size > no split).
        - Training set size limit: if train_size > 0, then limit the 'train' split size.
        """
        train = raw_dataset["train"]

        val_is_requested_by_ratio = self.config.val_ratio is not None and self.config.val_ratio > 0.0

        val_size_as_split_is_requested = self.config.val_size is not None and self.config.val_size > 0
        
        if val_is_requested_by_ratio:
            if val_size_as_split_is_requested:
                logger.warning(
                    "Both val_ratio and val_size are set. val_ratio will take precedence for splitting the validation set."
                )

            test_split_param = self.config.val_ratio
            logger.info(f"Splitting train dataset using val_ratio: {self.config.val_ratio}")
            
            split = train.train_test_split(test_size=test_split_param, seed=self.config.seed)
            train = split["train"]
            val = split["test"]

        elif val_size_as_split_is_requested:
            test_split_param = self.config.val_size
            logger.info(f"Splitting train dataset using absolute count val_size: {self.config.val_size}")
            split = train.train_test_split(test_size=test_split_param, seed=self.config.seed)
            train = split["train"]
            val = split["test"]

        else:
            val = train.select(range(0)) 
            logger.info("Validation split skipped as neither ratio nor size was requested.")
                
        # 2. Limit the dataset size (Train Size)
        if self.config.train_size is not None and self.config.train_size > 0:
            train_size = min(self.config.train_size, len(train))
            train = train.shuffle(seed=self.config.seed).select(range(train_size))
            logger.info(f"Train set limited to {train_size} samples.")
        else:
            logger.info("No limit applied to train set size.")

        return DatasetDict({"train": train, "validation": val})

    def _tokenize_dataset(self, processed_dataset: DatasetDict) -> DatasetDict:
        """
        Subclasses get tokenization function and apply it to train/val splits.
        """
        tokenize_fn = self._get_tokenize_fn() # Get tokenization function from subclass
        
        train = processed_dataset["train"]
        val = processed_dataset["validation"]

        # Determine whether to use batched map based on configuration
        is_batched = self.config.batched_map 

        # Tokenize Train Set
        train_tokenized = train.map(
            tokenize_fn,
            batched=is_batched,
            remove_columns=train.column_names,
            desc="Tokenizing train",
        )
        
        # Tokenize Validation Set (only operate on non-empty sets)
        if len(val) > 0:
            val_tokenized = val.map(
                tokenize_fn,
                batched=is_batched,
                remove_columns=val.column_names,
                desc="Tokenizing validation",
            )
        else:
            val_tokenized = val

        return DatasetDict({"train": train_tokenized, "validation": val_tokenized})

    def load_and_split(self) -> DatasetDict:
        """
        Public method: Load → Split → Tokenize.
        Returns tokenized DatasetDict with 'train' and 'validation' splits.
        """
        logger.info("Loading and tokenizing dataset...")
        self.raw_dataset = self._load_raw_data()
        processed = self._split_and_limit_dataset(self.raw_dataset)
        self.dataset = self._tokenize_dataset(processed)
        self._tokenized = True

        logger.info(
            f"Dataset ready: {len(self.dataset['train'])} train, "
            f"{len(self.dataset['validation'])} validation"
        )
        return self.dataset

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
    ) -> Optional[DataLoaderProtocol]:
        """ 
        Return a DataLoader for the specified split. 
        General logic for all dataset types.
        """
        self._ensure_tokenized()
        
        if split not in self.dataset:
            logger.warning(f"Split '{split}' not found in dataset. Returning None.")
            return None
            
        if len(self.dataset[split]) == 0:
            logger.warning(f"Split '{split}' found but is empty. Returning None.")
            return None
        
        dataset = self.dataset[split]

        if collate_fn is None:
            collate_fn = self._get_default_collate_fn()
        
        if 'labels' in dataset.column_names:
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        else:
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _ensure_tokenized(self) -> None:
        if not self._tokenized:
            raise RuntimeError(
                "Dataset not tokenized. Call load_and_split() first."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_length={self.config.max_length}, "
            f"cache_dir={self.config.cache_dir}, "
            f"tokenized={self._tokenized})"
        )