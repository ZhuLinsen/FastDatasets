import os
import sys
from loguru import logger
from dotenv import load_dotenv
from app.core.config import config

# 加载环境变量
load_dotenv()

# 配置日志
logger.remove()  # 移除默认的处理器

# 添加控制台输出
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=config.LOG_LEVEL
)

# 添加文件输出
log_file = config.LOG_FILE
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger.add(
    log_file,
    rotation=config.LOG_MAX_SIZE,  # 已在config.py中确保值为字符串
    retention=config.LOG_BACKUP_COUNT,
    format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",  # 使用固定格式，而不是配置文件中的格式字符串
    level=config.LOG_LEVEL,
    encoding="utf-8"
) 