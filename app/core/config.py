import os
from dotenv import load_dotenv

load_dotenv()

def parse_log_max_size(val):
    """
    解析日志文件大小配置，确保返回 loguru 兼容的字符串格式
    
    支持的输入格式:
    - "10MB" -> "10 MB"
    - "10 MB" -> "10 MB" (不变)
    - 10485760 -> "10 MB" (转换字节到MB)
    - "10485760" -> "10 MB" (转换字节字符串到MB)
    """
    if val is None:
        return "10 MB"
    
    # 把纯数字输入转成 "10 MB" 格式
    if isinstance(val, (int, float)) or (isinstance(val, str) and val.isdigit()):
        bytes_val = int(str(val).strip())
        mb_val = max(1, bytes_val // (1024 * 1024))  # 至少 1MB
        return f"{mb_val} MB"
    
    # 处理带单位的字符串
    val = str(val).strip().upper()
    
    # 添加空格 (如 "10MB" -> "10 MB")
    if val.endswith(("KB", "MB", "GB")) and " " not in val:
        for unit in ("KB", "MB", "GB"):
            if val.endswith(unit):
                size = val[:-len(unit)].strip()
                if size.isdigit():  # 确保是数字
                    return f"{size} {unit}"
    
    # 已经是正确格式 (如 "10 MB")
    for unit in ("KB", "MB", "GB"):
        if f" {unit}" in val:
            return val
    
    # 默认返回 10 MB
    return "10 MB"

class Config:
    # Chunk length range
    CHUNK_MIN_LEN = int(os.getenv("DOCUMENT_MIN_CHUNK_SIZE", "1500").split('#')[0].strip())
    CHUNK_MAX_LEN = int(os.getenv("DOCUMENT_MAX_CHUNK_SIZE", "2000").split('#')[0].strip())
    # LLM API config
    API_KEY = os.getenv("LLM_API_KEY", "your-api-key")
    BASE_URL = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    MODEL_NAME = os.getenv("LLM_MODEL", "your-model-name")
    # Input/output
    INPUT_PATH = os.getenv("STORAGE_PATH", "data/input/")
    OUTPUT_DIR = os.getenv("STORAGE_PATH", "data/output/")
    # Language
    LANGUAGE = os.getenv("LANGUAGE", "中文")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "你是一个有用的助手。")
    # Output formats
    OUTPUT_FORMATS = os.getenv("OUTPUT_FORMATS", "alpaca,sharegpt").split(",")
    OUTPUT_FILE_FORMAT = os.getenv("DATASET_FORMAT", "json")
    # Feature switches
    ENABLE_COT = os.getenv("ENABLE_COT", "False") == "True"
    ENABLE_LABEL = os.getenv("ENABLE_LABEL", "False") == "True"
    ENABLE_OPTIMIZE = os.getenv("ENABLE_OPTIMIZE", "True") == "True"
    MAX_LLM_CONCURRENCY = int(os.getenv("MAX_LLM_CONCURRENCY", 10))
    # API
    DATA_PATH = os.getenv("DATA_PATH", "data/datasets.json")
    API_PREFIX = "/api"
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    LOG_MAX_SIZE = parse_log_max_size(os.getenv("LOG_MAX_SIZE", "10 MB"))
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))
    # 文档处理设置
    DOCUMENT_EXTRACTION_TIMEOUT = int(os.getenv("TASK_TIMEOUT", 60))
    # 预留更多全局参数

config = Config() 