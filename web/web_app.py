#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastDatasets Web Interface
基于Gradio的数据集生成Web界面
"""

import os
import sys
import json
import threading
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import gradio as gr
from loguru import logger
from rich.console import Console

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from app.core.config import Config
    from app.core.document import DocumentProcessor
    from app.core.dataset import DatasetBuilder
    # from app.core.logger import setup_logger  # logger已在模块中配置
    from app.services.dataset_service import TextSplitter
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目根目录: {project_root}")
    print(f"Python路径: {sys.path[:3]}")
    sys.exit(1)

# 导入Web模块
from config import web_config
from utils import (
    format_file_size, format_duration, generate_task_id,
    validate_file_format, get_file_info, format_status_message,
    format_results_summary, clean_old_tasks, save_task_state,
    load_task_state, validate_config
)
from status_manager import status_manager

# 初始化
console = Console()
config = Config()
# setup_logger()  # logger已在模块导入时配置

# 全局变量
processing_status = load_task_state(web_config.PROJECT_ROOT / "web" / "tasks.json")
processing_results = load_task_state(web_config.PROJECT_ROOT / "web" / "results.json")
processing_lock = threading.Lock()

# 定期清理旧任务
def cleanup_old_tasks():
    """清理旧任务"""
    with processing_lock:
        clean_old_tasks(processing_status, 24)
        clean_old_tasks(processing_results, 24)
        save_task_state(processing_status, web_config.PROJECT_ROOT / "web" / "tasks.json")
        save_task_state(processing_results, web_config.PROJECT_ROOT / "web" / "results.json")

# 启动时恢复任务状态
def recover_zombie_tasks():
    """恢复僵尸任务 - 将长时间运行的任务标记为失败"""
    active_tasks = status_manager.get_active_tasks()
    now = datetime.now()
    
    for task_id, task in active_tasks.items():
        # 检查任务是否运行超过配置的超时时间
        elapsed = now - task.start_time
        if elapsed.total_seconds() > web_config.TASK_TIMEOUT:
            logger.warning(f"发现僵尸任务 {task_id}，运行时间: {elapsed}，将标记为失败")
            status_manager.fail_task(
                task_id, 
                f"任务超时（运行 {int(elapsed.total_seconds()//60)} 分钟）"
            )

# 启动时执行恢复
recover_zombie_tasks()

# 启动清理线程
cleanup_thread = threading.Thread(target=lambda: [time.sleep(3600), cleanup_old_tasks()], daemon=True)
cleanup_thread.start()

class WebInterface:
    """Web界面主类"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.dataset_generator = DatasetBuilder()
        self.text_splitter = TextSplitter()
        
    def upload_files(self, files) -> Tuple[str, str]:
        """处理文件上传"""
        if not files:
            return "❌ 请选择要上传的文件", ""
        # 兼容单文件上传
        if isinstance(files, str):
            files = [files]
        
        uploaded_files = []
        file_info_list = []
        
        for file_path in files:
            if file_path:
                # 验证文件格式
                if not validate_file_format(file_path, web_config.SUPPORTED_FORMATS):
                    file_ext = Path(file_path).suffix.lower()
                    return f"❌ 不支持的文件格式: {file_ext}\n支持的格式: {', '.join(web_config.SUPPORTED_FORMATS)}", ""
                
                # 获取文件信息
                file_info = get_file_info(file_path)
                if file_info:
                    uploaded_files.append(file_path)
                    file_info_list.append(f"📄 {file_info['name']} ({file_info['size_formatted']})")
                else:
                    return f"❌ 无法读取文件: {Path(file_path).name}", ""
        
        if not uploaded_files:
            return "❌ 没有有效的文件", ""
        
        file_list = "\n".join(file_info_list)
        total_size = sum(get_file_info(f).get('size', 0) for f in uploaded_files)
        
        return (f"✅ 成功上传 {len(uploaded_files)} 个文件 (总大小: {format_file_size(total_size)}):\n{file_list}", 
                json.dumps(uploaded_files))
    
    def start_processing_with_status(self, 
                        files_json: str,
                        output_dir: str,
                        chunk_min_len: int,
                        chunk_max_len: int,
                        output_formats: List[str],
                        enable_cot: bool,
                        llm_concurrency: int,
                        file_concurrency: int,
                        api_key: str,
                        base_url: str,
                        model_name: str,
                        questions_per_chunk: int = 2) -> str:
        """开始处理文档并返回状态信息"""
        result = self.start_processing(
            files_json, output_dir, chunk_min_len, chunk_max_len,
            output_formats, enable_cot, llm_concurrency, file_concurrency,
            api_key, base_url, model_name, questions_per_chunk
        )
        
        # 如果成功开始处理，返回详细状态
        if result.startswith("✅"):
            return self.get_processing_status()
        else:
            return f"## 处理失败\n\n{result}"
    
    def start_processing(self, 
                        files_json: str,
                        output_dir: str,
                        chunk_min_len: int,
                        chunk_max_len: int,
                        output_formats: List[str],
                        enable_cot: bool,
                        llm_concurrency: int,
                        file_concurrency: int,
                        api_key: str,
                        base_url: str,
                        model_name: str,
                        questions_per_chunk: int = 2) -> str:
        """开始处理文档"""
        
        logger.info(f"收到处理请求，文件列表: {files_json}")
        
        if not files_json:
            return "❌ 请先上传文件"
        
        try:
            files = json.loads(files_json)
        except:
            return "❌ 文件信息解析失败"
        
        if not files:
            return "❌ 没有可处理的文件"
        
        # 验证配置参数
        config_dict = {
            "chunk_min_len": chunk_min_len,
            "chunk_max_len": chunk_max_len,
            "output_formats": output_formats,
            "llm_concurrency": llm_concurrency,
            "file_concurrency": file_concurrency
        }
        
        is_valid, error_msg = validate_config(config_dict)
        if not is_valid:
            return f"❌ 配置验证失败: {error_msg}"
        
        # 生成任务ID
        task_id = generate_task_id()
        
        # 更新配置
        config.CHUNK_MIN_LEN = chunk_min_len
        config.CHUNK_MAX_LEN = chunk_max_len
        config.OUTPUT_FORMATS = output_formats
        config.ENABLE_COT = enable_cot
        config.MAX_LLM_CONCURRENCY = llm_concurrency
        config.MAX_FILE_CONCURRENCY = file_concurrency
        
        if api_key:
            config.API_KEY = api_key
        if base_url:
            config.BASE_URL = base_url
        if model_name:
            config.MODEL_NAME = model_name
        
        # 设置输出目录
        if not output_dir:
            output_dir = str(project_root / "output")
        
        # 使用新的状态管理器创建任务
        status_manager.create_task(task_id, len(files))
        
        # 启动异步处理
        def run_async_task():
            import asyncio
            try:
                logger.info(f"开始异步处理 {len(files)} 个文件 - 任务ID: {task_id}")
                
                # 简化事件循环管理
                asyncio.run(self._process_files_async(task_id, files, output_dir, questions_per_chunk))
                logger.info(f"任务 {task_id} 处理完成")
                    
            except Exception as e:
                logger.error(f"异步处理失败: {e}")
                import traceback
                logger.error(f"详细错误信息: {traceback.format_exc()}")
                try:
                    status_manager.fail_task(task_id, str(e))
                except Exception as status_error:
                    logger.error(f"错误状态更新失败: {status_error}")
        
        thread = threading.Thread(
            target=run_async_task,
            daemon=True
        )
        thread.start()
        
        return f"✅ 开始处理任务 {task_id}，共 {len(files)} 个文件"
    
    async def _process_files_async(self, task_id: str, files: List[str], output_dir: str, questions_per_chunk: int = 2):
        """异步处理文件"""
        logger.info(f"开始异步处理 {len(files)} 个文件 - 任务ID: {task_id}")
        
        # 初始化导出文件列表
        export_files = []
        
        # 获取输出格式配置
        output_formats = config.OUTPUT_FORMATS or ["alpaca"]
        
        try:
            # 阶段1: 文档处理
            logger.info(f"更新任务状态 - 任务ID: {task_id}")
            status_manager.update_task(task_id, 
                status="processing",
                stage="文档解析", 
                message="正在解析和分块文档...",
                stage_progress=0
            )
            logger.info(f"状态更新完成 - 任务ID: {task_id}")
            
            results = []
            total_files = len(files)
            
            for i, file_path in enumerate(files):
                logger.info(f"处理文件 {i+1}/{total_files}: {Path(file_path).name}")
                
                # 更新当前处理文件
                try:
                    status_manager.update_task(task_id,
                        current_file=Path(file_path).name,
                        progress=int((i / total_files) * 30),  # 文档处理占30%
                        stage_progress=int((i / total_files) * 100),
                        message=f"正在处理文档 {i+1}/{total_files}: {Path(file_path).name}"
                    )
                    logger.info(f"文件状态更新完成 - 任务ID: {task_id}")
                except Exception as status_error:
                    logger.error(f"状态更新失败: {status_error}")
                
                try:
                    # 处理单个文件
                    logger.info(f"开始处理文档: {file_path}")
                    result = self.doc_processor.process_document(file_path)
                    logger.info(f"文档处理完成，得到 {len(result) if result else 0} 个文档块")
                    
                    if result:
                        results.extend(result)
                    
                    # 更新已处理文件数
                    try:
                        status_manager.update_task(task_id, processed_files=i + 1)
                        logger.info(f"已处理文件数更新: {i + 1}")
                    except Exception as status_error:
                        logger.error(f"已处理文件数更新失败: {status_error}")
                        
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    
                    try:
                        status_manager.update_task(task_id, 
                            message=f"处理文件 {Path(file_path).name} 失败: {str(e)}"
                        )
                    except Exception as status_error:
                        logger.error(f"错误状态更新失败: {status_error}")
            
            logger.info(f"文档处理完成，共得到 {len(results)} 个文档块")
            
            # 阶段2: 生成数据集
            if results:
                total_chunks = len(results)
                logger.info(f"开始生成数据集，共 {total_chunks} 个文档块")
                
                # 阶段2.1: 生成问题
                try:
                    status_manager.update_task(
                        task_id,
                        stage="生成问题",
                        message=f"正在为 {total_chunks} 个文档块生成问题...",
                        progress=35,
                        stage_progress=0,
                        total_chunks=total_chunks,
                        processed_chunks=0
                    )
                    logger.info(f"生成问题阶段状态更新完成")
                except Exception as status_error:
                    logger.error(f"生成问题阶段状态更新失败: {status_error}")
                
                # 导出数据集
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"输出目录已创建: {output_path}")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 保存原始数据
                raw_file = output_path / f"raw_data_{timestamp}.json"
                try:
                    with open(raw_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logger.info(f"原始数据已保存: {raw_file}")
                except Exception as e:
                    logger.error(f"保存原始数据失败: {e}")
                
                # 生成问答对
                qa_pairs = []
                try:
                    # 阶段2.2: 生成答案
                    try:
                        status_manager.update_task(
                            task_id,
                            stage="生成答案",
                            message="正在生成问题答案...",
                            progress=60,
                            stage_progress=0
                        )
                        logger.info(f"生成答案阶段状态更新完成")
                    except Exception as status_error:
                        logger.error(f"生成答案阶段状态更新失败: {status_error}")
                    
                    # 初始化 DatasetBuilder
                    logger.info(f"开始生成问答对...")
                    dataset_generator = DatasetBuilder()
                    qa_pairs = await dataset_generator.build_dataset(results)
                    logger.info(f"问答对生成完成，共 {len(qa_pairs)} 个")
                    
                    # 阶段2.3: 保存结果
                    try:
                        status_manager.update_task(
                            task_id,
                            stage="保存结果",
                            message="正在保存结果...",
                            progress=85,
                            stage_progress=0
                        )
                        logger.info(f"保存结果阶段状态更新完成")
                    except Exception as status_error:
                        logger.error(f"保存结果阶段状态更新失败: {status_error}")
                    
                except Exception as e:
                    logger.error(f"生成问答对失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    
                    try:
                        status_manager.update_task(
                            task_id,
                            message=f"生成问答对失败: {str(e)}"
                        )
                    except Exception as status_error:
                        logger.error(f"问答对失败状态更新失败: {status_error}")
                
                # 阶段3: 导出文件
                try:
                    status_manager.update_task(
                        task_id,
                        stage="保存结果",
                        message="正在导出数据集文件...",
                        progress=95,
                        stage_progress=0
                    )
                    logger.info(f"导出文件阶段状态更新完成")
                except Exception as status_error:
                    logger.error(f"导出文件阶段状态更新失败: {status_error}")
                
                # 导出不同格式
                try:
                    # 使用DatasetBuilder的export_dataset方法
                    logger.info(f"开始导出数据集文件...")
                    dataset_generator.export_dataset(
                        qa_pairs, 
                        str(output_path), 
                        output_formats, 
                        "json"
                    )
                    logger.info(f"数据集导出完成")
                    
                    # 构建导出文件列表
                    for fmt in output_formats:
                        file_path = output_path / f"dataset-{fmt}.json"
                        if file_path.exists():
                            export_files.append(str(file_path))
                            logger.info(f"导出文件确认: {file_path}")
                        else:
                            logger.warning(f"导出文件不存在: {file_path}")
                    
                    logger.info(f"数据集已导出到: {output_path}")
                except Exception as e:
                    logger.error(f"导出数据集失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                
                # 完成处理
                try:
                    logger.info(f"标记任务完成 - 任务ID: {task_id}")
                    # 先更新qa_pairs字段
                    status_manager.update_task(task_id, qa_pairs=len(qa_pairs))
                    status_manager.complete_task(
                        task_id,
                        message=f"处理完成！生成了 {len(qa_pairs)} 个问答对"
                    )
                    logger.info(f"任务完成状态更新成功 - 任务ID: {task_id}")
                    
                    # 保存结果到全局变量
                    processing_results[task_id] = {
                        "qa_pairs": len(qa_pairs),
                        "export_files": export_files,
                        "raw_file": str(raw_file)
                    }
                    logger.info(f"结果已保存到全局变量")
                    
                except Exception as status_error:
                    logger.error(f"任务完成状态更新失败: {status_error}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    
            else:
                logger.warning(f"没有成功处理任何文档")
                try:
                    status_manager.fail_task(
                        task_id,
                        "没有成功处理任何文档"
                    )
                    logger.info(f"失败状态更新成功")
                except Exception as status_error:
                    logger.error(f"失败状态更新失败: {status_error}")
                    
        except Exception as e:
            logger.error(f"处理任务 {task_id} 发生严重错误: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            try:
                status_manager.fail_task(
                    task_id,
                    f"处理失败: {str(e)}"
                )
                logger.info(f"严重错误状态更新成功")
            except Exception as status_error:
                logger.error(f"严重错误状态更新失败: {status_error}")
                
        finally:
            logger.info(f"任务 {task_id} 处理流程结束")
    
    def get_processing_status(self) -> str:
        """获取处理状态"""
        summary = status_manager.get_summary()
        active_tasks = status_manager.get_active_tasks()
        
        if not active_tasks and summary['total_tasks'] == 0:
            return "暂无处理任务"

        # 当前任务详情
        if summary['current_task']:
            task = summary['current_task']
            
            # 计算运行时间
            elapsed = datetime.now() - task.start_time
            elapsed_str = f"{int(elapsed.total_seconds()//60)}分{int(elapsed.total_seconds()%60)}秒"
            
            # 简化状态映射
            stage_map = {
                '初始化': '开始处理',
                '文档解析': '开始处理', 
                '文本分块': '开始处理',
                '生成问题': '生成问题中',
                '生成答案': '生成答案中',
                '保存结果': '完成',
                '任务完成': '完成'
            }
            
            display_stage = stage_map.get(task.stage, task.stage)
            
            # 根据任务状态显示不同信息
            if task.status == 'completed':
                # 获取生成的问答对数量
                qa_count = getattr(task, 'qa_pairs', 0) or task.processed_chunks * 2  # 估算
                return f"**状态**: 完成，共生成 {qa_count} 个问答对\n**处理时间**: {elapsed_str}"
            elif task.status == 'failed':
                return f"**状态**: 处理失败\n**处理时间**: {elapsed_str}\n**错误**: {task.error or '未知错误'}"
            else:
                return f"**状态**: {display_stage}\n**处理时间**: {elapsed_str}"
        
        return "暂无活跃任务"
    
    def get_results_info(self) -> str:
        """获取结果信息"""
        with processing_lock:
            if not processing_results:
                return "📋 暂无完成的任务"
            
            results_text = "## 处理结果\n\n"
            
            for task_id, result in processing_results.items():
                results_text += f"### 任务 {task_id}\n"
                results_summary = format_results_summary(result)
                results_text += f"{results_summary}\n"
                results_text += "---\n\n"
            
            # 保存结果到文件
            save_task_state(processing_results, web_config.PROJECT_ROOT / "web" / "results.json")
            
            return results_text
    
    def export_datasets(self, export_formats: List[str]) -> List[str]:
        """导出已完成的数据集并返回文件路径列表"""
        try:
            # 从状态管理器获取已完成任务
            completed = status_manager.get_completed_tasks()
            if not completed:
                return []
            
            # 优先选择"已完成"的任务（排除失败的），否则回退到所有结束的任务
            completed_only = {k: v for k, v in completed.items() if v.status == 'completed'}
            source_pool = completed_only if completed_only else completed
            
            # 选择最新完成的任务（按 end_time）
            latest_task_id, latest_task = max(
                source_pool.items(), key=lambda item: item[1].end_time or datetime.min
            )
            
            # 任务完成时，_process_files_async 会在输出目录下生成 dataset-alpaca.json / dataset-sharegpt.json
            # 需要根据 raw_file 推断输出目录
            # 先尝试从 processing_results 中找到对应信息作为回退
            original_output_dir = None
            with processing_lock:
                latest_result = processing_results.get(latest_task_id)
                if latest_result and latest_result.get('raw_file'):
                    original_output_dir = Path(latest_result['raw_file']).parent
            
            # 如果未能通过 processing_results 找到，使用默认输出目录
            if original_output_dir is None:
                original_output_dir = Path(web_config.DEFAULT_OUTPUT_DIR)
            
            # 使用默认导出目录
            export_dir = Path("./exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            exported_files = []
            for fmt in export_formats:
                try:
                    # 匹配 DatasetBuilder.export_dataset 的命名: dataset-alpaca.json / dataset-sharegpt.json
                    file_name = f"dataset-{fmt}.json"
                    source_file = original_output_dir / file_name
                    if source_file.exists():
                        import shutil
                        dest_file = export_dir / f"{latest_task_id[:8]}_{source_file.name}"
                        shutil.copy2(source_file, dest_file)
                        exported_files.append(str(dest_file))
                    else:
                        logger.warning(f"未找到导出文件: {source_file}")
                except Exception as e:
                    logger.error(f"导出 {fmt} 时出错: {e}")
                    continue
            
            return exported_files
        except Exception as e:
            logger.error(f"导出数据集时出错: {e}")
            return []

# 创建Web界面实例
web_interface = WebInterface()

def auto_refresh_smart(prev_status: str, prev_results: str):
    """智能定时刷新：仅在内容变化时更新，减少闪烁
    通过归一化忽略"用时"字段的秒级变化，避免每次tick都触发渲染。
    """
    import re

    def normalize_status(s: str) -> str:
        if not s:
            return ""
        # 将 "**用时**: X分Y秒" 替换为固定占位，避免秒级抖动
        s = re.sub(r"\*\*用时\*\*: [^\|\n]+", "**用时**: —", s)
        return s.strip()

    new_status = web_interface.get_processing_status()
    new_results = web_interface.get_results_info()

    update_status = normalize_status(new_status) != normalize_status(prev_status)
    update_results = new_results != prev_results

    out_status = new_status if update_status else prev_status
    out_results = new_results if update_results else prev_results

    # 返回：显示用的状态与结果，以及用于下次比较的最新状态与结果
    return out_status, out_results, new_status, new_results

def auto_refresh():
    """定时刷新处理状态和结果"""
    return web_interface.get_processing_status(), web_interface.get_results_info()

# 创建Gradio界面
def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(
        title="FastDatasets - 智能数据集生成工具",
        theme=web_config.get_gradio_theme(),
        css=web_config.get_custom_css()
    ) as interface:
        
        # 标题和说明
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """
                    <div style="text-align: center; padding: 20px;">
                        <h1 style="color: #2563eb; margin-bottom: 10px;">FastDatasets</h1>
                        <p style="color: #64748b; font-size: 18px; margin-bottom: 5px;">智能数据集生成工具</p>
                        <p style="color: #94a3b8; font-size: 14px;">将文档转换为高质量的LLM训练数据集</p>
                    </div>
                    """
                )
        
        # ===== 主功能区采用 2×2 网格布局 =====
        # 第一行：上传文件 + 参数设置
        with gr.Row():
            # 左列：上传文件
            with gr.Column(scale=1):
                with gr.Accordion("📤 上传文件", open=True):
                    with gr.Row():
                        file_upload = gr.Files(
                            label="📤 选择文档文件 (支持: PDF, DOCX, TXT, MD)",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            scale=2
                        )
                        
                        upload_status = gr.Textbox(
                            label="上传状态",
                            interactive=False,
                            lines=3,
                            placeholder="请选择要处理的文档文件...",
                            scale=1
                        )
                        
                        files_data = gr.Textbox(
                            visible=False,
                            value=""
                        )
            
            # 右列：参数设置
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ 参数设置", open=False):
                    with gr.Row():
                        output_dir = gr.Textbox(
                            label="📁 输出目录",
                            value=web_config.DEFAULT_OUTPUT_DIR,
                            placeholder="留空使用默认目录",
                            scale=2
                )
                
                        chunk_min_len = gr.Number(
                            label="最小块长度",
                            value=web_config.DEFAULT_CHUNK_MIN_LEN,
                            minimum=50,
                            maximum=10000,
                            scale=1
                )
                
                        chunk_max_len = gr.Number(
                            label="最大块长度",
                            value=web_config.DEFAULT_CHUNK_MAX_LEN,
                            minimum=500,
                            maximum=20000,
                            scale=1
                )
                
                        questions_per_chunk = gr.Number(
                            label="问题数/块",
                            value=2,
                            minimum=1,
                            maximum=10,
                            scale=1
                )
        
                    # 第二行参数
                    with gr.Row():
                        output_formats = gr.CheckboxGroup(
                            label="📋 输出格式",
                            choices=["alpaca", "sharegpt"],
                            value=web_config.DEFAULT_OUTPUT_FORMATS,
                            scale=1
                        )
                
                        enable_cot = gr.Checkbox(
                            label="🧠 启用思维链 (CoT)",
                            value=web_config.DEFAULT_ENABLE_COT,
                            scale=1
                        )
                
                        llm_concurrency = gr.Number(
                            label="LLM并发数",
                            value=web_config.DEFAULT_LLM_CONCURRENCY,
                            minimum=1,
                            maximum=50,
                            scale=1
                        )
                
                        file_concurrency = gr.Number(
                            label="文件并发数",
                            value=web_config.DEFAULT_FILE_CONCURRENCY,
                            minimum=1,
                            maximum=20,
                            scale=1
                        )
        
            # LLM配置区域 - 横向布局
            with gr.Row():
                api_key = gr.Textbox(
                label="🔑 API Key",
                type="password",
                placeholder="留空使用环境变量中的配置",
                scale=2
            )
            
                base_url = gr.Textbox(
                label="🌐 Base URL",
                placeholder="留空使用环境变量中的配置",
                scale=2
            )
            
                model_name = gr.Textbox(
                label="🤖 模型名称",
                placeholder="留空使用环境变量中的配置",
                scale=1
            )
            
                start_btn = gr.Button(
                "🚀 开始处理",
                variant="primary",
                size="lg",
                scale=1
            )
        
        # 第二行：处理状态 和 数据下载
        with gr.Row():
            # 左列：处理状态
            with gr.Column(scale=1):
                with gr.Accordion("📈 处理状态", open=True):
                    status_display = gr.Markdown(
                        "📋 等待任务开始...",
                        elem_classes=["progress-section"]
                    )
        
            # 右列：数据下载
            with gr.Column(scale=1):
                with gr.Accordion("📥 数据下载", open=False):
                    with gr.Row():
                        export_format = gr.CheckboxGroup(
                            choices=["alpaca", "sharegpt"],
                            value=["alpaca"],
                            label="📂 导出格式",
                            scale=1
                        )
            
                    export_btn = gr.Button(
                        "📥 导出数据集",
                        variant="primary",
                        scale=1
                    )
            
                    downloaded_files = gr.Files(
                        label="下载导出的文件",
                        interactive=False
                    )
        
        # 处理结果模块已移除

        # 状态存储
        status_state = gr.State("")

        
        # 事件绑定
        file_upload.change(
            fn=web_interface.upload_files,
            inputs=[file_upload],
            outputs=[upload_status, files_data]
        )
        
        # 开始处理按钮事件
        start_btn.click(
            fn=web_interface.start_processing_with_status,
            inputs=[
                files_data, output_dir, chunk_min_len, chunk_max_len,
                output_formats, enable_cot, llm_concurrency, file_concurrency,
                api_key, base_url, model_name, questions_per_chunk
            ],
            outputs=[status_display]
        )
        
        # 定时刷新状态
        refresh_timer = gr.Timer(value=2.0)
        refresh_timer.tick(
            fn=lambda prev_status: web_interface.get_processing_status(),
            inputs=[status_state],
            outputs=[status_display]
        )
        
        # 导出按钮事件
        export_btn.click(
            fn=web_interface.export_datasets,
            inputs=[export_format],
            outputs=[downloaded_files]
        )
    
    return interface

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='FastDatasets Web Interface')
    parser.add_argument('--port', type=int, default=web_config.PORT, help='服务端口号')
    parser.add_argument('--host', type=str, default=web_config.HOST, help='服务主机地址')
    parser.add_argument('--share', action='store_true', help='启用公网访问')
    args = parser.parse_args()
    
    # 创建并启动界面
    interface = create_interface()
    
    # 启动服务
    # 优先使用命令行参数，其次是环境变量，最后是配置文件
    server_port = args.port or int(os.getenv("GRADIO_SERVER_PORT", web_config.PORT))
    server_host = args.host or web_config.HOST
    share = args.share or web_config.SHARE
    
    interface.launch(
        server_name=server_host,
        server_port=server_port,
        share=share,
        show_error=True,
        quiet=False,
        inbrowser=web_config.INBROWSER,
        show_api=False
    )