#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web界面工具函数
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """格式化时间间隔"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} 分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} 小时"

def generate_task_id() -> str:
    """生成任务ID"""
    timestamp = str(int(time.time() * 1000))
    random_str = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"task_{timestamp}_{random_str}"

def validate_file_format(file_path: str, supported_formats: List[str]) -> bool:
    """验证文件格式"""
    if not file_path:
        return False
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_formats

def get_file_info(file_path: str) -> Dict[str, Any]:
    """获取文件信息"""
    if not file_path or not os.path.exists(file_path):
        return {}
    
    file_stat = os.stat(file_path)
    file_path_obj = Path(file_path)
    
    return {
        "name": file_path_obj.name,
        "size": file_stat.st_size,
        "size_formatted": format_file_size(file_stat.st_size),
        "extension": file_path_obj.suffix.lower(),
        "modified_time": datetime.fromtimestamp(file_stat.st_mtime),
        "path": str(file_path_obj.absolute())
    }

def create_progress_bar(current: int, total: int, width: int = 30) -> str:
    """创建文本进度条"""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = int(progress * 100)
    
    return f"[{bar}] {percentage}% ({current}/{total})"

def format_status_message(status: Dict[str, Any]) -> str:
    """格式化状态消息"""
    STATUS_EMOJI = {
        "starting": "",
        "processing": "",
        "completed": "",
        "failed": "",
    }
    
    icon = STATUS_EMOJI.get(status.get("status", "unknown"), "")
    
    # 计算运行时间
    start_time = status.get("start_time")
    if start_time:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        elapsed = datetime.now() - start_time
        elapsed_str = format_duration(elapsed.total_seconds())
    else:
        elapsed_str = "未知"
    
    # 创建进度条
    current = status.get("processed_files", 0)
    total = status.get("total_files", 0)
    progress_bar = create_progress_bar(current, total)
    
    message = f"""{icon} **{status.get('status', '未知')}**

**进度**: {progress_bar}
**当前文件**: {status.get('current_file', '无')}
**运行时间**: {elapsed_str}
**消息**: {status.get('message', '无')}
"""
    
    return message

def format_results_summary(results: Dict[str, Any]) -> str:
    """格式化结果摘要"""
    qa_pairs = results.get("qa_pairs", 0)
    export_files = results.get("export_files", [])
    raw_file = results.get("raw_file", "")
    
    summary = f"""**生成统计**:
- 问答对数量: {qa_pairs} 个
- 导出文件: {len(export_files)} 个

**文件列表**:
"""
    
    if raw_file:
        summary += f"- 原始数据: `{Path(raw_file).name}`\n"
    
    for file_path in export_files:
        file_name = Path(file_path).name
        file_size = format_file_size(os.path.getsize(file_path)) if os.path.exists(file_path) else "未知"
        summary += f"- {file_name} ({file_size})\n"
    
    return summary

def clean_old_tasks(task_dict: Dict[str, Any], max_age_hours: int = 24) -> None:
    """清理旧任务"""
    current_time = datetime.now()
    tasks_to_remove = []
    
    for task_id, task_data in task_dict.items():
        start_time = task_data.get("start_time")
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            
            age = current_time - start_time
            if age > timedelta(hours=max_age_hours):
                tasks_to_remove.append(task_id)
    
    for task_id in tasks_to_remove:
        del task_dict[task_id]

def save_task_state(task_dict: Dict[str, Any], file_path: str) -> None:
    """保存任务状态到文件"""
    try:
        # 转换datetime对象为字符串
        serializable_dict = {}
        for task_id, task_data in task_dict.items():
            serializable_data = task_data.copy()
            if "start_time" in serializable_data and isinstance(serializable_data["start_time"], datetime):
                serializable_data["start_time"] = serializable_data["start_time"].isoformat()
            serializable_dict[task_id] = serializable_data
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存任务状态失败: {e}")

def load_task_state(file_path: str) -> Dict[str, Any]:
    """从文件加载任务状态"""
    try:
        if not os.path.exists(file_path):
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            task_dict = json.load(f)
        
        # 转换字符串为datetime对象
        for task_id, task_data in task_dict.items():
            if "start_time" in task_data and isinstance(task_data["start_time"], str):
                task_data["start_time"] = datetime.fromisoformat(task_data["start_time"])
        
        return task_dict
    except Exception as e:
        print(f"加载任务状态失败: {e}")
        return {}

def create_download_link(file_path: str) -> str:
    """创建文件下载链接"""
    if not os.path.exists(file_path):
        return "文件不存在"
    
    file_name = Path(file_path).name
    file_size = format_file_size(os.path.getsize(file_path))
    
    return f"📎 [{file_name}]({file_path}) ({file_size})"

def validate_config(config_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """验证配置参数"""
    errors = []
    
    # 验证数值范围
    if "chunk_min_len" in config_dict:
        if not (50 <= config_dict["chunk_min_len"] <= 10000):
            errors.append("最小块长度必须在50-10000之间")
    
    if "chunk_max_len" in config_dict:
        if not (500 <= config_dict["chunk_max_len"] <= 20000):
            errors.append("最大块长度必须在500-20000之间")
    
    if "llm_concurrency" in config_dict:
        if not (1 <= config_dict["llm_concurrency"] <= 50):
            errors.append("LLM并发数必须在1-50之间")
    
    if "file_concurrency" in config_dict:
        if not (1 <= config_dict["file_concurrency"] <= 20):
            errors.append("文件并发数必须在1-20之间")
    
    # 验证输出格式
    if "output_formats" in config_dict:
        valid_formats = ["alpaca", "sharegpt"]
        for fmt in config_dict["output_formats"]:
            if fmt not in valid_formats:
                errors.append(f"不支持的输出格式: {fmt}")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "配置验证通过"

def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    import psutil
    
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
    }