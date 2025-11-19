from transformers import PreTrainedTokenizer, default_data_collator, DataCollatorForLanguageModeling
from typing import Callable, Dict, Any, List
import logging
import json

from .base_dataset import BaseDataset, logger
from ..models.trainer_config import DatasetConfig


# logger = logging.getLogger(__name__)

class TextInstructionDataset(BaseDataset):
    """
    A dataset that requires fine-tuning (SFT) and specific labeling for question answering tasks.
    Suitable for tasks such as SQuAD, Instruction Tuning, and Chat Format.
    """
    # _logged_first_sample = False

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        config.batched_map = True
        super().__init__(config, tokenizer)
        self._logged_first_sample = False

    def _get_answer_text(self, ex: Dict[str, Any]) -> str:
        if ex.get("answers"): # SQuAD style
            answer_text = ex.get("answers", {}).get("text", [])
            return answer_text[0] if len(answer_text) > 0 else "The question cannot be answered from the context."
        elif ex.get("answer"): # Instruction Tuning style
            return ex.get("answer")
        elif ex.get("output"): # COIG / Alpaca style
            return ex.get("output")   
        elif ex.get("response"): # Chat Format style
            return ex.get("response")
        else:
            return None

    def _build_default_messages(self, context, question, system_prompt, answer):
        if context:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
                {"role": "assistant", "content": answer}
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "assistant", "content": answer}
            ]

    def _process_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:

        batch_size = len(next(iter(batch.values())))
        messages_full_batch = []
        messages_prompt_only_batch = []
        
        # 1. Python 迴圈：創建所有樣本的 messages 列表
        for i in range(batch_size):
            ex = {k: batch[k][i] for k in batch.keys()}
            
            messages_full = ex.get("messages", None)
            if not (messages_full and isinstance(messages_full, list)):
                context = ex.get("context", "")
                question = ex.get("question", ex.get("instruction", ""))
                
                if not question:
                    raise ValueError(f"Sample {i}: Dataset must contain a 'question' or 'instruction' column.")

                answer_text = self._get_answer_text(ex)
                if answer_text is None:
                    raise ValueError(
                        f"Sample {i}: Must contain one of the following columns: 'answers', 'answer', 'output', or 'response'."
                    )

                # 決定 System Prompt
                system_prompt = self.config.default_system_prompt or ex.get("system_prompt", "")
                if not system_prompt:
                    if context:
                        system_prompt = "Answer the question using only the information in the context. If the answer is not mentioned, say 'The question cannot be answered from the context'."
                    else:
                        system_prompt = "Answer the question as best as you can."
                
                # 構建消息列表
                messages_full = self._build_default_messages(
                    context, question, system_prompt, answer_text
                )

            messages_prompt_only = messages_full[:-1]
            messages_full_batch.append(messages_full)
            messages_prompt_only_batch.append(messages_prompt_only)

        # 2. 批次分詞 (Batch Tokenization) - 效率核心
        
        # 完整 messages：用於 input_ids, attention_mask
        tokenized_full = self.tokenizer.apply_chat_template(
            conversation=messages_full_batch, 
            truncation=True, 
            max_length=self.config.max_length, 
            padding="max_length", # 填充到 max_length
            return_tensors=None, # 返回 List[List[int]]
            return_dict=True,
        )
        
        # Prompt only：用於計算真實 Prompt 長度
        prompt_only = self.tokenizer.apply_chat_template(
            conversation=messages_prompt_only_batch,
            truncation=True, 
            padding=False, # 不填充
            return_tensors=None,
            add_generation_prompt=True,
            return_dict=True,
        )

        # 3. Python 迴圈：應用 Label Masking (必須逐個樣本處理長度)
        final_labels = []
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        
        for i in range(batch_size):
            full_ids = tokenized_full['input_ids'][i]
            len_prompt_tokens = len(prompt_only['input_ids'][i])
            current_labels = full_ids.copy()
            
            # 3.1. 屏蔽 Prompt 部分
            current_labels[:len_prompt_tokens] = [-100] * len_prompt_tokens
            start_search_index = len_prompt_tokens
            # 3.2. 屏蔽 Padding 部分
            try:
                first_pad_index = current_labels.index(pad_id, start_search_index)
                full_ids[first_pad_index] = eos_id
                current_labels[first_pad_index] = eos_id
                current_labels[first_pad_index + 1:] = [-100] * (len(current_labels) - (first_pad_index + 1))
            except ValueError:
                pass

            tokenized_full['input_ids'][i] = full_ids
            final_labels.append(current_labels)
            
            # 4. 調試日誌 (只記錄第一個樣本)
            if i == 0 and not self._logged_first_sample:
                logger.info(f"--- Tokenizing First Sample (Batch Mode) ---")
                messages_full = messages_full_batch[i] # 獲取第一個樣本的 messages
                logger.info(f"Raw messages_full: {json.dumps(messages_full, ensure_ascii=False, indent=2)}")
                
                tokens_full = self.tokenizer.convert_ids_to_tokens(full_ids)
                debug_output = []
                max_log_tokens = 100 
                
                for input_id, label, token in zip(full_ids[:max_log_tokens], final_labels[i][:max_log_tokens], tokens_full[:max_log_tokens]):
                    is_answer = '✅' if label != -100 else '❌' 
                    debug_output.append(f"{is_answer} Label: {label:5d}, Token: {repr(token)}")

                logger.info(f"Token Lengths: Full={len(full_ids)}, Prompt={len_prompt_tokens}")
                logger.info(f"Labels Masking (Input/Label Alignment, showing first {max_log_tokens} tokens):")
                logger.info("\n".join(debug_output))
                self._logged_first_sample = True


        tokenized_full["labels"] = final_labels
        return tokenized_full

    def _tokenize_single_fallback(self, ex: Dict[str, Any]) -> Dict[str, List[int]]:
        messages_full = ex.get("messages", None)
        if not (messages_full and isinstance(messages_full, list)):
            context = ex.get("context", "")
            question = ex.get("question", "")
            system_prompt = ex.get("system_prompt", "")

            if self.config.default_system_prompt:
                system_prompt = self.config.default_system_prompt

            # === Fallback for COIG / Alpaca style ===
            if not question and ex.get("instruction"):
                question = ex["instruction"]
            if not context and ex.get("input"):
                context = ex["input"]

            if not question:
                raise ValueError("Dataset must contain a 'question' column.")
            
            answer_text = self._get_answer_text(ex)
            if answer_text is None:
                raise ValueError(
                    f"Must contain one of the following columns: 'answers', 'answer', 'output', or 'response'."
                )
            
            if not system_prompt:
                if context:
                    system_prompt = (
                        "Answer the question using only the information in the context. "
                        "If the answer is not mentioned, say 'The question cannot be answered from the context'."
                    )
                else:
                    system_prompt = "Answer the question as best as you can."

            messages_full = self._build_default_messages(
                context, question, system_prompt, answer_text
            )

        messages_prompt_only = messages_full[:-1]
        
        tokenized_full = self.tokenizer.apply_chat_template(
            conversation=messages_full, 
            truncation=True, 
            max_length=self.config.max_length, 
            padding="max_length",
            return_dict=True,
        )
        
        prompt_only = self.tokenizer.apply_chat_template(
            conversation=messages_prompt_only,
            truncation=True, 
            max_length=self.config.max_length, 
            padding=False,
            return_dict=True,
        )
        
        len_prompt_tokens = len(prompt_only["input_ids"])
        
        labels = tokenized_full['input_ids'].copy()
        labels[:len_prompt_tokens] = [-100] * len_prompt_tokens 

        pad_id = self.tokenizer.pad_token_id 
        try:
            try:
                start_index = len_prompt_tokens
                first_pad_index = labels.index(pad_id, start_index)
            except ValueError:
                first_pad_index = len(labels)
                
            labels[first_pad_index:] = [-100] * (len(labels) - first_pad_index)

        except Exception as e:
            logger.error(f"Error during padding masking: {e}")
            pass

        if not self._logged_first_sample:
            logger.info(f"--- Tokenizing First Sample ---")
            logger.info(f"Raw messages_full: {json.dumps(messages_full, ensure_ascii=False, indent=2)}")
            tokens_full = self.tokenizer.convert_ids_to_tokens(tokenized_full["input_ids"])
            
            # 組合 input_ids, labels 和 tokens_full 成一個列表方便對齊查看
            debug_output = []
            max_log_tokens = 100 # 限制日誌長度
            
            for input_id, label, token in zip(tokenized_full["input_ids"][:max_log_tokens], labels[:max_log_tokens], tokens_full[:max_log_tokens]):
                # 標籤不是 -100 (即答案部分) 的部分會被標記
                is_answer = '✅' if label != -100 else '❌' 
                debug_output.append(f"{is_answer} Label: {label:4d}, Token: {repr(token)}") # 使用 repr 確保特殊 token 可見

            logger.info(f"Token Lengths: Full={len(tokenized_full['input_ids'])}, Prompt={len_prompt_tokens}")
            logger.info(f"Labels Masking (Input/Label Alignment, showing first {max_log_tokens} tokens):")
            logger.info("\n".join(debug_output))
            
            # 設定旗標，之後不再列印
            self._logged_first_sample = True

        return {
            "input_ids": tokenized_full["input_ids"],
            "attention_mask": tokenized_full["attention_mask"],
            "labels": labels
        }

    def _get_tokenize_fn(self) -> Callable:
        def tokenize_fn(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            try:
                # 嘗試高效的批次處理
                return self._process_batch(batch)
            except Exception as e:
                logger.warning(f"Batch tokenization failed: {e}. Falling back to single-sample tokenization.")
                
                batch_size = len(next(iter(batch.values())))
                
                # 執行單樣本回退
                outputs = [
                    self._tokenize_single_fallback({k: batch[k][i] for k in batch.keys()})
                    for i in range(batch_size)
                ]
                
                # 將單樣本結果重新組合成批次格式
                return {
                    "input_ids": [o["input_ids"] for o in outputs],
                    "attention_mask": [o["attention_mask"] for o in outputs],
                    "labels": [o["labels"] for o in outputs],
                }
        return tokenize_fn

    def _get_default_collate_fn(self) -> Callable:
        return default_data_collator
