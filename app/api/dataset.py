from fastapi import APIRouter, HTTPException, Body
from app.services.dataset_service import DatasetService
import asyncio
from app.core.logger import logger
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ChunkModel(BaseModel):
    content: str
    summary: Optional[str] = None
    
class QAPair(BaseModel):
    question: str
    answer: str
    
class QAPairList(BaseModel):
    qa_pairs: List[QAPair]
    
class ExportRequest(BaseModel):
    data: List[Dict[str, Any]]

router = APIRouter()
service = DatasetService()

@router.post("/split")
def split_documents_api(input_path: str = None):
    """拆分文档并返回文本块"""
    chunks = service.split_documents(input_path)
    return {"chunks": chunks}

@router.post("/generate")
async def generate_qa_api(chunk: ChunkModel, number: int = 5):
    """
    为文本块生成问答对
    
    Args:
        chunk: 文本块（至少包含content字段）
        number: 生成的问题数量
    
    Returns:
        生成的问答对列表
    """
    try:
        # 将ChunkModel转换为字典
        chunk_dict = chunk.dict()
        
        # 确保有content字段
        if not chunk_dict.get('content'):
            raise HTTPException(status_code=400, detail="文本块必须包含content字段")
            
        logger.info(f"为文本块生成问题，文本长度: {len(chunk_dict['content'])}, 问题数量: {number}")
        
        questions = await service.generate_questions(chunk_dict, number=number)
        
        # 为每个问题生成答案
        qa_pairs = []
        for question in questions:
            logger.info(f"正在为问题生成答案: {question[:30]}...")
            answer = await service.generate_answer(chunk_dict, question)
            qa_pairs.append({"question": question, "answer": answer})
        
        return {"qa_pairs": qa_pairs}
    except Exception as e:
        logger.error(f"生成问答对时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成问答对失败: {str(e)}")

@router.post("/optimize")
async def optimize_qa_api(qa_pairs: List[Dict[str, str]]):
    """
    优化问答对
    
    Args:
        qa_pairs: 问答对列表，每个包含question和answer字段
    
    Returns:
        优化后的问答对
    """
    try:
        if not qa_pairs:
            raise HTTPException(status_code=400, detail="问答对列表不能为空")
            
        optimized_pairs = []
        for pair in qa_pairs:
            if "answer" not in pair:
                logger.warning(f"跳过没有answer字段的问答对: {pair}")
                continue
                
            optimized = await service.optimize_answer(pair["answer"])
            optimized_pairs.append({
                "question": pair.get("question", ""),
                "answer": optimized
            })
        
        return {"optimized_pairs": optimized_pairs}
    except Exception as e:
        logger.error(f"优化问答对时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"优化问答对失败: {str(e)}")

@router.post("/export")
def export_data_api(request: ExportRequest = Body(...), output_path: str = None, file_format: str = "json"):
    """
    导出数据到文件
    
    Args:
        request: 包含要导出的数据列表
        output_path: 输出文件路径
        file_format: 文件格式，默认为json
    
    Returns:
        导出结果
    """
    try:
        if not output_path:
            output_path = f"output/dataset-{file_format}.{file_format}"
            
        service.export_data(request.data, output_path, file_format)
        return {"status": "success", "message": f"数据已成功导出到 {output_path}", "format": file_format}
    except Exception as e:
        logger.error(f"导出数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"导出数据失败: {str(e)}") 