from fastapi import APIRouter, HTTPException, Query, Path, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union

from ..core.training_job_manager import get_job_manager
from ..training.models.job_submission import TrainingJobSubmission


router = APIRouter(
    prefix="/training",
    tags=["AI 模型訓練任務管理 (Training Job Management)"]
)

def get_manager():
    return get_job_manager()

# --- Pydantic Models for Requests and Responses ---

class JobStatusResponse(BaseModel):
    """單個訓練任務的狀態響應模型。"""
    job_id: str = Field(..., description="唯一的任務 ID")
    gpu_id: Union[int, List[int]] = Field(None, description="分配到的 GPU ID")
    status: str = Field(..., description="任務狀態 (queued, running, completed, failed, cancelled)")
    config: Optional[Dict[str, Any]] = Field(None, description="訓練配置詳情")
    start_time: Optional[str] = Field(None, description="開始時間 (ISO 8601)")
    end_time: Optional[str] = Field(None, description="結束時間 (ISO 8601)")
    metrics: Optional[Dict[str, Any]] = Field(None, description="最終訓練指標或錯誤信息")
    model_path: Optional[str] = Field(None, description="訓練完成後模型的存儲路徑")
# --- API Endpoints ---

@router.post("/submit", response_model=JobStatusResponse, summary="提交一個新的訓練任務")
def submit_training_job_endpoint(
    submission: TrainingJobSubmission,
    job_manager = Depends(get_manager)
):
    """
    提交一個新的 AI 模型訓練任務到排程器。

    ---

    對於Instruction Tuning / SFT 任務，系統支援以多種欄位組合提供Instruction Dataset，所有欄位會依照
    `column_mapping` 自動轉換為標準化的 Chat Messages 格式，以便用於
    Instruction Tuning / SFT 訓練。

    **支援欄位（依照 column_mapping 定義）：**
    - `messages`：完整多輪對話（若存在將直接使用）
    - `reasoning`：模型思考過程（CoT）
    - `context`：上下文
    - `input`：指令、問題或提示文字（如 prompt、instruction、question）
    - `output`：模型期望回答（如 response、answer）

    **單輪資料自動轉換格式：**
    若資料未提供 `messages` 欄位，系統會自動組成以下對話：
    - system: 系統的 `default_system_prompt` 或資料內的 `system_prompt`
    - user:
        - 若有 `context` → `Context: {context} Question/Instruction: {input}`
        - 否則 → `{input}`
    - assistant:
        - 若提供 `reasoning` → 會合併為:

            <think>
            {reasoning}
            </think>

            {output}

        - 否則直接為 `{output}`

    **多輪訊息（messages）格式要求：**
    - 每個項目需包含：`{"role": "system|user|assistant", "content": str}`
    - 會自動清洗空內容或無效 role
    - 最後一個 assistant 回覆會自動補上 EOS token（若 tokenizer 有定義）

    ---

    對於 Text Generation 任務，一樣可透過 `column_mapping` 指定 `text` 欄位來提供純文字生成資料。
    """
    try:
        job_id = job_manager.submit_job(submission)
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=500, detail="Job submission failed internally.")
            
        return job.to_dict() # 返回 job 的詳細信息
        
    except Exception as e:
        # 注意: job_manager.submit_job 是同步的，但可能會拋出內部錯誤
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@router.get("/status/{job_id}", response_model=JobStatusResponse, summary="獲取特定任務的詳細狀態")
async def get_job_status_endpoint(
    job_id: str = Path(..., description="要查詢的任務 ID"),
    job_manager = Depends(get_manager)
):
    """
    查詢指定 ID 訓練任務的當前狀態、分配的 GPU 和結果。
    """
    # 1. 嘗試獲取即時進度（這會執行 MLflow 查詢）
    progress_data = job_manager.get_job_progress(job_id)

    # 2. 獲取 Job 的基本/最終狀態 (來自 Redis)
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    
    response_data = job.to_dict()
    if job.status != "completed" and progress_data:
        if response_data["metrics"] is None:
            response_data["metrics"] = {}
        response_data["metrics"].update(progress_data)
        
    return response_data


@router.get("/list", response_model=List[JobStatusResponse], summary="列出所有訓練任務")
def list_all_jobs_endpoint(
    status: Optional[str] = Query(None, description="按狀態過濾 (queued, running, completed, failed, cancelled)"),
    job_manager = Depends(get_manager)
):
    """
    列出所有已提交的訓練任務，可選按狀態過濾。
    """
    try:
        jobs_list = job_manager.list_jobs(status=status)
        
        # 將簡化列表轉換為完整的 JobStatusResponse 列表
        full_jobs_list = []
        for job_summary in jobs_list:
            # 載入完整 Job 對象以匹配 JobStatusResponse
            job = job_manager.get_job(job_summary['job_id'])
            progress_data = job_manager.get_job_progress(job_summary['job_id'])

            response_data = job.to_dict()
            if job.status != "completed" and progress_data:
                if response_data["metrics"] is None:
                    response_data["metrics"] = {}
                response_data["metrics"].update(progress_data) 

            full_jobs_list.append(response_data)
            
        return full_jobs_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.post("/cancel/{job_id}", summary="取消一個正在運行或排隊中的訓練任務")
def cancel_job_endpoint(
    job_id: str = Path(..., description="要取消的任務 ID"),
    job_manager = Depends(get_manager)
):
    """
    嘗試取消指定的訓練任務。
    
    - 如果任務在排隊中 (queued)，將直接移除。
    - 如果任務在運行中 (running)，將標記為 'cancelled'，訓練器會在下一個檢查點自行終止。
    """
    success = job_manager.cancel_job(job_id)
    if not success:
        # 如果任務不存在或狀態已經是 completed/failed/cancelled
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
        
        if job.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(status_code=400, detail=f"Job '{job_id}' is already in status: {job.status}. Cannot cancel.")

        raise HTTPException(status_code=500, detail="Failed to cancel job for unknown reason.")
        
    return {"message": f"Cancellation request processed for Job ID: {job_id}"}


@router.delete("/delete/{job_id}", summary="刪除任務記錄")
def delete_job_endpoint(
    job_id: str = Path(..., description="要刪除的任務 ID"),
    force: bool = Query(False, description="如果任務仍在運行或排隊中，是否強制取消並刪除"),
    job_manager = Depends(get_manager)
):
    """
    從 Redis 和備份中刪除任務記錄和模型。
    
    - 如果任務正在運行或排隊中，必須設定 `force=true` 才能刪除。
    """
    try:
        deleted = job_manager.delete_job(job_id, force_cancel=force)
        if deleted:
            return {"message": f"Job ID '{job_id}' successfully deleted."}
        else:
            job = job_manager.get_job(job_id)
            if job and job.status in ["running", "queued"] and not force:
                raise HTTPException(status_code=400, detail=f"Job '{job_id}' is still {job.status}. Use 'force=true' to cancel and delete.")
            
            raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found or could not be deleted.")
    except HTTPException:
        # 重新拋出 400 錯誤
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during deletion: {str(e)}")