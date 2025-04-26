import json
from typing import Any, Dict, List
from app.core.config import config
from app.core.logger import logger
import os

class JSONStorage:
    def __init__(self, path: str = None):
        self.path = path or config.DATA_PATH

    def load(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"{self.path} not found, returning empty list.")
            return []
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return []

    def save(self, data: List[Dict[str, Any]]):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save data: {e}")

    def export(self, data: List[Dict[str, Any]], output_path: str, file_format: str = "json"):
        """
        导出数据到文件
        Args:
            data: 要导出的数据
            output_path: 输出路径
            file_format: 文件格式，json或jsonl
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 确保文件扩展名正确
        base_path, ext = os.path.splitext(output_path)
        if file_format == "json" and ext != ".json":
            output_path = base_path + ".json"
        elif file_format == "jsonl" and ext != ".jsonl":
            output_path = base_path + ".jsonl"
            
        # 根据格式导出
        if file_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            # 默认使用json格式，确保是正确的JSON数组格式
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

storage = JSONStorage() 