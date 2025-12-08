# RAPTOR Training Framework API Technical Documentation

This document guides developers on how to manage resources, submit fine-tuning jobs, and monitor training workflows via the REST API.

---

## 📚 Table of Contents
1.  [Best Practices](#1-best-practices)
2.  [Resource Management API](#2-resource-management-api)
3.  [Core: Submit Job](#3-core-submit-job)
    *   [3.1 Root Level Parameters](#31-root-level-parameters-job-submission)
    *   [3.2 Training Config Details](#32-training-config-details)
        *   [General & Hyperparameters](#a-general-settings--hyperparameters)
        *   [Optimizer & Precision](#b-optimizer--precision-settings)
        *   [LoRA Settings](#c-lora-settings)
        *   [Quantization Settings (QLoRA)](#d-quantization-settings)
        *   [DeepSpeed Settings](#e-deepspeed-settings)
    *   [3.3 Dataset Config Details](#33-dataset-config-details)
        *   [Source & Splitting](#a-source--splitting)
        *   [Column Mapping](#b-column-mapping---core-feature)
4.  [Training Lifecycle Management API](#4-training-lifecycle-management-api)
5.  [Dashboards](#5-dashboards)
6.  [FAQ & Troubleshooting](#6-faq--troubleshooting)

---

## 1. Best Practices

It is highly recommended to refer to the [**`train_with_raptor_api.py`**](../test_trainer/train_with_raptor_api.py) script in the project. This script implements a simple automated workflow:

1.  **Check & Download**: Call the Resource APIs first to ensure the model and dataset are downloaded to the Server's local cache (`/tmp/...`).
2.  **Compose Config**: Use the **absolute local paths** returned by the Resource APIs to assemble the training configuration. This prevents re-downloading or network timeouts during training.
3.  **Submit**: Send the JSON payload to `/training/submit`.
4.  **Poll**: Poll `/training/status/{job_id}` until the status becomes `completed` or `failed`.

---

## 2. Resource Management API

Before starting training, the base model and data must be prepared.

### 2.1 Download Model
*   **Endpoint**: `POST /models/download`
*   **Purpose**: Download model weights from Hugging Face to the server's shared storage.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model_source` | string | Yes | Fixed as `"huggingface"`. |
| `model_name` | string | Yes | Hugging Face ID, e.g., `"google/gemma-3-270m-it"`. |

**Response (Success)**:
```json
{
  "message": "模型下載請求已處理",
  "details": "/app/tmp/models/google_gemma-3-270m-it"
}
```
> 💡 **Key**: Save the returned `model_path` to fill in `model_name_or_path` in the training config later.

### 2.2 Download Dataset
*   **Endpoint**: `POST /datasets/download_from_network`
*   **Purpose**: Download and cache the dataset.

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `dataset_source` | string | Yes | Fixed as `"huggingface"`. |
| `dataset_name` | string | Yes | HF Dataset ID, e.g., `"yentinglin/TaiwanChat"`. |

**Response (Success)**:
```json
{
  "success": true,
  "message": "數據集下載成功",
  "local_path": "/app/tmp/datasets/yentinglin_TaiwanChat",
  "dataset_source": "huggingface",
  "dataset_name": "yentinglin/TaiwanChat",
  ...
}
```
> 💡 **Key**: Save the returned `local_path` to fill in `cache_dir` in the dataset config later.

---

## 3. Core: Submit Job

*   **Endpoint**: `POST /training/submit`
*   **Header**: `Content-Type: application/json`

This section contains the most complex parameters. The request body consists of three main parts: Root, Training Config, and Dataset Config. Corresponds to the `TrainingJobSubmission` in [`job_submission.py`](../src/training/models/job_submission.py).

### 3.1 Root Level Parameters (Job Submission)
Controls hardware allocation and task type.

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `task_type` | string | Yes | - | Task type.<br>• `"instruction"`: For Chat, QA, SFT.<br>• `"text"`: For pure text completion (Pre-training). |
| `select_multiple_gpus` | bool | Yes | `false` | Enable multi-card training.<br>• `true`: Attempts to allocate multiple GPUs and start DDP/DeepSpeed.<br>• `false`: Uses a single GPU only. |
| `vram_budget_gb` | float | Yes | - | **Scheduling Threshold**. Estimated VRAM (GB) required.<br>The scheduler looks for GPUs where `Available VRAM > This Value`. If resources are insufficient, the job enters `queued` status. |
| `training_config` | object | Yes | - | Training details (See 3.2). |
| `dataset_config` | object | Yes | - | Dataset details (See 3.3). |

---

### 3.2 Training Config Details

Corresponds to the `TrainerConfig` dataclass in [`trainer_config.py`](../src/training/models/trainer_config.py).

#### A. General Settings & Hyperparameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name_or_path` | string | **Required** | Model path. Strongly recommended to use the **absolute path** returned by the Resource API. |
| `experiment_name` | string | `"llm-training"` | MLflow experiment name, used to group multiple runs. |
| `run_name` | string | `null` | MLflow Run name. Defaults to Job ID if left empty. |
| `max_epochs` | int | `3` | Max training epochs. |
| `batch_size` | int | `1` | Batch size per **Single GPU**. |
| `gradient_accumulation_steps` | int | `4` | Steps to accumulate gradients.<br>Effective Batch Size = `batch_size` × `accum_steps` × `num_gpus`.<br>Increase this if VRAM is tight. |
| `learning_rate` | float | `2e-5` | Learning rate. |
| `weight_decay` | float | `0.01` | Weight decay to prevent overfitting. |
| `warmup_steps` | int | `100` | Number of warmup steps. |
| `warmup_ratio` | float | `null` | Warmup ratio (e.g., `0.03`). Overrides `warmup_steps` if set. |
| `logging_steps` | int | `10` | Log to MLflow/TensorBoard every N steps. |
| `val_check_interval` | float | `1.0` | Validation frequency.<br>`1.0`: Validate at end of epoch.<br>`0.5`: Validate every half epoch. |

#### B. Optimizer & Precision Settings
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `use_bfloat16` | bool | `true` | Use `bf16` precision. Recommended for Ampere (A100/3090) or newer GPUs. |
| `use_flash_attn` | bool | `false` | Enable Flash Attention 2. Requires compatible hardware and libraries. |
| `use_8bit_adamw` | bool | `false` | Use 8-bit AdamW optimizer. **Highly recommended** to save VRAM. |
| `gradient_checkpointing` | bool | `false` | Enable Gradient Checkpointing.<br>**Recommended**. Slightly slows down training (~20%) but significantly reduces VRAM usage (~50-60%). |
| `max_grad_norm` | float | `1.0` | Gradient clipping threshold to prevent exploding gradients. |

#### C. LoRA Settings
If this object is not empty, PEFT/LoRA mode is automatically enabled.

| Parameter | Type | Suggested | Description |
| :--- | :--- | :--- | :--- |
| `r` | int | `8` or `16` | LoRA Rank. Higher values mean more parameters and better fitting but higher VRAM usage. |
| `lora_alpha` | int | `16` or `32` | Scaling factor, usually set to 2x `r`. |
| `lora_dropout` | float | `0.05` | Dropout probability. |
| `target_modules` | list[str] | `["q_proj", "v_proj"]` | Layers to apply LoRA.<br>For best results, include all linear layers: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. |
| `bias` | string | `"none"` | Whether to train bias terms. Usually `"none"`. |

#### D. Quantization Settings
Used for QLoRA training. Must be used with `lora_config`.

| Parameter | Type | Suggested | Description |
| :--- | :--- | :--- | :--- |
| `load_in_4bit` | bool | `true` | Load base model in 4-bit. |
| `bnb_4bit_quant_type` | string | `"nf4"` | Quantization type. `nf4` (Normal Float 4) is generally better than `fp4`. |
| `bnb_4bit_use_double_quant` | bool | `true` | Enable double quantization to save a bit more VRAM. |
| `bnb_4bit_compute_dtype` | string | `"bfloat16"` | Dequantization precision during computation. Should match `use_bfloat16`. |

#### E. DeepSpeed Settings
The system supports both simplified parameters and full JSON config.

**Method 1: Simplified Parameters**
*   `deepspeed_stage` (int): `2` (Optimizer Partition) or `3` (Parameter Partition).
*   `deepspeed_offload_optimizer` (bool): Offload optimizer state to CPU RAM.
*   `deepspeed_offload_parameters` (bool): (Stage 3 only) Offload model parameters to CPU RAM.

**Method 2: Full JSON (Recommended)**
*   `deepspeed_config`: Pass a standard DeepSpeed JSON object. If set, simplified parameters are ignored.

---

### 3.3 Dataset Config Details

Corresponds to `DatasetConfig` in `trainer_config.py` and logic in `TextInstructionDataset.py`.

#### A. Source & Splitting
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `dataset_name_or_path` | string | **Required**. Dataset path (HF ID or JSON file path). |
| `cache_dir` | string | **Highly Recommended**. Fill with `local_path` from Resource API to specify cache location. |
| `max_length` | int | `2048`. Max token length (Prompt + Output). Truncated if exceeded. |
| `train_split_name` | string | `"train"`. Name of the training split in the HuggingFace Dataset. |
| `val_split_name` | string | `null`. Name of the validation split. If missing, it will be split automatically based on rules below. |
| `val_ratio` | float | `null` (e.g., 0.1). **Ratio** to split from train set for validation. Priority > `val_size`. |
| `val_size` | int | `null` (e.g., 100). **Count** to split from train set for validation. |
| `train_size` | int | `null`. Limit the number of training samples (for debugging). |

#### B. Column Mapping - **Core Feature**
Determines how the system reads and converts your data. The system automatically converts data into standard conversation format.

| Mapping Key | Your Column Name | Logic |
| :--- | :--- | :--- |
| `messages` | e.g., `"conversations"` | **Highest Priority**. Use if your data is already in message format (`[{"role": "user"...}, ...]`). |
| `input` | e.g., `"question"` | **Single-turn**. Maps to user instruction/input. |
| `output` | e.g., `"answer"` | **Single-turn**. Maps to model expected response. |
| `context` | e.g., `"passage"` | (Optional) Passage or background information the model should reference before answering the question. . Combined as: `Context: ...\n\nQuestion: ...` |
| `reasoning` | e.g., `"cot"` | (Optional) Chain of Thought. Automatically wrapped as: `<think>{reasoning}</think>\n\n{output}`. |

**Example (TaiwanChat Dataset):**
```json
"column_mapping": {
    "messages": "conversations"
}
```
*Logic: First looks for `conversations`. If not found, looks for `instruction` + `output` to assemble.*

---

## 4. Training Lifecycle Management API

### 4.1 Get Job Status
*   **Endpoint**: `GET /training/status/{job_id}`
*   **Response (Running)**:
    ```json
    {
      "job_id": "job_123",
      "status": "running",
      "metrics": {
         "progress_percentage": 15.2,
         "train_loss_step": 0.85,
         "estimated_time_remaining_seconds": 1200
      },
      "config": { ... }
    }
    ```
*   **Response (Completed)**: `metrics` will contain `final_model_path`.

### 4.2 List Jobs
*   **Endpoint**: `GET /training/list`
*   **Query**: `?status=queued` (Optional)

### 4.3 Cancel Job
*   **Endpoint**: `POST /training/cancel/{job_id}`
*   **Behavior**: Graceful Shutdown. The training process saves a checkpoint and stops after the current step finishes.

### 4.4 Delete Job
*   **Endpoint**: `DELETE /training/delete/{job_id}`
*   **Query**: `?force=true` (Required if the job is stuck and cannot be cancelled normally).
*   **Behavior**: Removes records from Redis and deletes Logs and Checkpoints from the disk.

---

## 5. Dashboards

*   **Swagger UI**: [`http://192.168.157.167:8009/docs`](http://192.168.157.167:8009/docs).
*   **MLflow UI**: [`http://192.168.157.167:5000`](http://192.168.157.167:5000) (View Loss curves, compare experiments).
*   **Redis**: [`http://192.168.157.123:5540`](http://192.168.157.123:5540) (View underlying job data).
*   **TensorBoard**: [`http://192.168.157.167:6006`](http://192.168.157.167:6006) (Alternative visualization).

---

## 6. FAQ & Troubleshooting

**Q1: Why is the job status stuck in `queued`?**
*   **Reason**: The scheduler found insufficient available VRAM.
*   **Formula**: `Total GPU VRAM - Used VRAM - System Buffer (1GB) > Your vram_budget_gb`.
*   **Solution**: Check if other jobs are running or try reducing `vram_budget_gb`.

**Q2: I'm getting OOM (Out of Memory) errors. How should I adjust parameters?**
Try the following adjustments in order:
1.  Enable `gradient_checkpointing: true` (Most effective).
2.  Enable `use_8bit_adamw: true`.
3.  Reduce `batch_size` (e.g., from 2 to 1) and double `gradient_accumulation_steps` (to keep effective batch size constant).
4.  Enable DeepSpeed.
5.  Use QLoRA.

