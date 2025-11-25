import argparse
import json
import sys
import os
import traceback
import logging
from pathlib import Path

sys.path.append(os.getcwd()) 

from src.training.models.trainer_config import TrainerConfig, DatasetConfig
from src.core.training_orchestrator import TrainingOrchestrator
from src.core.training_job_manager import CancelToken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (TrainScript) - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--redis_url", type=str, required=True)
    parser.add_argument("--redis_password", type=str, default="")
    parser.add_argument("--mlflow_uri", type=str, required=True)
    
    args = parser.parse_args()

    try:
        # 1. 讀取配置
        with open(args.config_path, "r") as f:
            job_config = json.load(f)

        trainer_config = TrainerConfig(**job_config["trainer_config"])
        dataset_config = DatasetConfig(**job_config["dataset_config"])
        task_type = job_config["task_type"]

        logger.info(f"Starting job {args.job_id} task {task_type}")

        # 2. 初始化 Cancel Token
        token = CancelToken(args.job_id, args.redis_url, args.redis_password)

        # 3. 啟動 Orchestrator
        orchestrator = TrainingOrchestrator(
            task_type=task_type,
            config=trainer_config,
            dataset_config=dataset_config,
            cancel_token=token,
            mlflow_uri=args.mlflow_uri,
        )

        metrics = orchestrator.run()

        # 4. 寫入成功結果
        result = {
            "status": "completed",
            "metrics": metrics,
            "final_model_path": metrics.get("final_model_path")
        }

    except Exception as e:
        logger.error(f"Training script failed: {e}", exc_info=True)
        result = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    # 5. 將結果寫回 JSON 檔案，供主進程讀取
    with open(args.result_path, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()