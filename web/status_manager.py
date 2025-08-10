#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态管理器
用于管理处理任务的状态和进度
"""

import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from loguru import logger

@dataclass
class TaskStatus:
    """任务状态数据类"""
    task_id: str
    status: str  # starting, processing, completed, failed
    stage: str  # 当前阶段
    progress: int  # 总体进度 0-100
    stage_progress: int  # 阶段进度 0-100
    total_files: int
    processed_files: int
    total_chunks: int
    processed_chunks: int
    current_file: str
    message: str
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    qa_pairs: int = 0  # 生成的问答对数量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理datetime对象
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskStatus':
        """从字典创建"""
        # 处理datetime字符串
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and isinstance(data['end_time'], str):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)

class StatusManager:
    """状态管理器"""
    
    def __init__(self, status_file: Path = None):
        self._tasks: Dict[str, TaskStatus] = {}
        # 使用可重入锁，避免同一线程重复获取锁导致死锁
        self._lock = threading.RLock()
        self._status_file = status_file or Path('/home/mumu/codes/FastDatasets/web/tasks_status.json')
        self._load_status()
        logger.debug(f"StatusManager initialized. status_file={self._status_file}")
    
    def create_task(self, task_id: str, total_files: int = 0) -> TaskStatus:
        """创建新任务"""
        with self._lock:
            task = TaskStatus(
                task_id=task_id,
                status="starting",
                stage="初始化",
                progress=0,
                stage_progress=0,
                total_files=total_files,
                processed_files=0,
                total_chunks=0,
                processed_chunks=0,
                current_file="",
                message="任务初始化中...",
                start_time=datetime.now()
            )
            self._tasks[task_id] = task
            self._save_status()
            logger.info(f"创建任务: {task_id}, total_files={total_files}")
            return task
    
    def update_task(self, task_id: str, **kwargs) -> Optional[TaskStatus]:
        """更新任务状态"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"update_task: 未找到任务 {task_id}")
                return None
            
            task = self._tasks[task_id]
            updated_keys = list(kwargs.keys())
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            # 自动计算进度（如果调用方未显式提供progress）
            if task.total_files > 0 and 'progress' not in kwargs:
                try:
                    file_progress = (task.processed_files / task.total_files) * 100
                    task.progress = int(file_progress)
                except Exception as e:
                    logger.debug(f"自动计算进度异常: {e}")
            
            self._save_status()
            logger.debug(f"更新任务 {task_id}: keys={updated_keys}, status={task.status}, stage={task.stage}, progress={task.progress}% stage_progress={task.stage_progress}% current_file={task.current_file}")
            return task
    
    def complete_task(self, task_id: str, message: str = "任务完成") -> Optional[TaskStatus]:
        """完成任务"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"complete_task: 未找到任务 {task_id}")
                return None
            
            task = self._tasks[task_id]
            task.status = "completed"
            task.stage = "完成"
            task.progress = 100
            task.stage_progress = 100
            task.message = message
            task.end_time = datetime.now()
            
            self._save_status()
            logger.info(f"任务完成: {task_id}")
            return task
    
    def fail_task(self, task_id: str, error: str) -> Optional[TaskStatus]:
        """任务失败"""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"fail_task: 未找到任务 {task_id}")
                return None
            
            task = self._tasks[task_id]
            task.status = "failed"
            task.stage = "失败"
            task.message = f"任务失败: {error}"
            task.error = error
            task.end_time = datetime.now()
            
            self._save_status()
            logger.error(f"任务失败: {task_id}, error={error}")
            return task
    
    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        with self._lock:
            task = self._tasks.get(task_id)
            logger.debug(f"get_task({task_id}) -> {bool(task)})")
            return task
    
    def get_all_tasks(self) -> Dict[str, TaskStatus]:
        """获取所有任务"""
        with self._lock:
            logger.debug(f"get_all_tasks() -> {len(self._tasks)} tasks")
            return self._tasks.copy()
    
    def get_active_tasks(self) -> Dict[str, TaskStatus]:
        """获取活跃任务"""
        with self._lock:
            active = {k: v for k, v in self._tasks.items() 
                   if v.status in ['starting', 'processing']}
            logger.debug(f"get_active_tasks() -> {len(active)} tasks")
            return active
    
    def get_completed_tasks(self) -> Dict[str, TaskStatus]:
        """获取已完成任务"""
        with self._lock:
            completed = {k: v for k, v in self._tasks.items() 
                   if v.status in ['completed', 'failed']}
            logger.debug(f"get_completed_tasks() -> {len(completed)} tasks")
            return completed
    
    def clean_old_tasks(self, hours: int = 24):
        """清理旧任务"""
        with self._lock:
            now = datetime.now()
            to_remove = []
            
            for task_id, task in self._tasks.items():
                if task.end_time and (now - task.end_time).total_seconds() > hours * 3600:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
            
            if to_remove:
                self._save_status()
                logger.info(f"清理旧任务: {to_remove}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取状态摘要（避免在锁内再次调用加锁方法）"""
        with self._lock:
            total_count = len(self._tasks)
            active_tasks = {k: v for k, v in self._tasks.items() if v.status in ['starting', 'processing']}
            completed_tasks = {k: v for k, v in self._tasks.items() if v.status in ['completed', 'failed']}
            
            summary = {
                'total_tasks': total_count,
                'active_tasks': len(active_tasks),
                'completed_tasks': len([t for t in completed_tasks.values() if t.status == 'completed']),
                'failed_tasks': len([t for t in completed_tasks.values() if t.status == 'failed']),
                'current_task': None
            }
            
            # 获取当前正在处理的任务（优先processing，其次starting）
            processing_tasks = [t for t in active_tasks.values() if t.status == 'processing']
            starting_tasks = [t for t in active_tasks.values() if t.status == 'starting']
            if processing_tasks:
                summary['current_task'] = processing_tasks[0]
            elif starting_tasks:
                summary['current_task'] = starting_tasks[0]
            
            logger.debug(f"get_summary() -> total={summary['total_tasks']}, active={summary['active_tasks']}, completed={summary['completed_tasks']}, failed={summary['failed_tasks']}")
            return summary
    
    def _save_status(self):
        """保存状态到文件"""
        try:
            data = {k: v.to_dict() for k, v in self._tasks.items()}
            with open(self._status_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"状态已保存: {self._status_file} ({len(self._tasks)} tasks)")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
    
    def _load_status(self):
        """从文件加载状态"""
        try:
            if self._status_file.exists():
                with open(self._status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._tasks = {k: TaskStatus.from_dict(v) for k, v in data.items()}
                logger.debug(f"已从文件加载任务状态: {self._status_file} ({len(self._tasks)} tasks)")
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
            self._tasks = {}

# 全局状态管理器实例
status_manager = StatusManager(Path("/home/mumu/codes/FastDatasets/web/tasks_status.json"))