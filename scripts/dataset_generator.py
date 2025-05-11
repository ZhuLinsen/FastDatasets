import os
import sys
import asyncio
import argparse
from pathlib import Path
from app.core.logger import logger
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetBuilder
from app.core.config import config
from tqdm.asyncio import tqdm as tqdm_async
import concurrent.futures
import json

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

async def process_document(file_path: str, output_dir: str = None, output_format: str = None):
    """处理单个文档"""
    try:
        # 初始化处理器
        processor = DocumentProcessor()
        builder = DatasetBuilder()
        
        # 解析文档
        logger.info(f"开始处理文档: {file_path}")
        
        # 分割文档
        chunks = processor.process_document(
            file_path,
            chunk_size=config.CHUNK_MAX_LEN,
            chunk_overlap=200
        )
        
        if not chunks:
            logger.error(f"文档处理失败: {file_path}")
            return
        
        # 生成数据集
        dataset = await builder.build_dataset(chunks)
        
        # 保存数据集
        if output_dir:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置输出文件名
            output_file_format = output_format or getattr(config, 'OUTPUT_FILE_FORMAT', 'jsonl')
            output_path = Path(output_dir) / f"{Path(file_path).stem}_dataset.{output_file_format}"
            
            # 保存原始数据集
            builder.save_dataset(dataset, output_path)
            logger.info(f"数据集已保存: {output_path}")
            
            # 导出多种格式（如 alpaca、sharegpt）
            if hasattr(config, 'OUTPUT_FORMATS') and config.OUTPUT_FORMATS:
                builder.export_dataset(
                    dataset, 
                    output_dir, 
                    config.OUTPUT_FORMATS, 
                    output_file_format
                )
                
        return dataset
        
    except Exception as e:
        logger.error(f"处理文档失败 {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

async def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理文档并生成数据集")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("-o", "--output", help="输出目录路径", default="./output")
    parser.add_argument("-f", "--format", help="输出格式(alpaca或sharegpt)", 
                        default=getattr(config, 'OUTPUT_FORMAT', 'alpaca'))
    parser.add_argument("-c", "--concurrency", help="并发处理数量", 
                        type=int, default=getattr(config, 'MAX_LLM_CONCURRENCY', 10))
    parser.add_argument("--file-format", help="输出文件格式(json或jsonl)", 
                        default=getattr(config, 'OUTPUT_FILE_FORMAT', 'jsonl'))
    parser.add_argument("--file-concurrency", help="文件并发处理数量", 
                        type=int, default=5)
    args = parser.parse_args()
    
    # 更新配置
    if args.format:
        os.environ['OUTPUT_FORMAT'] = args.format
        config.OUTPUT_FORMAT = args.format
    
    if args.concurrency:
        os.environ['MAX_LLM_CONCURRENCY'] = str(args.concurrency)
        config.MAX_LLM_CONCURRENCY = args.concurrency
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 处理输入路径
    input_path = Path(args.input)
    if input_path.is_file():
        # 处理单个文件
        await process_document(str(input_path), args.output, args.file_format)
    elif input_path.is_dir():
        # 收集目录中的所有文件
        file_paths = []
        for file_path in input_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
                file_paths.append(str(file_path))
        
        if file_paths:
            logger.info(f"找到 {len(file_paths)} 个文件待处理，开始批量处理...")
            
            # 设置最大并发数
            max_concurrent = min(args.file_concurrency, 8)  # 限制文件处理的并发数
            sem = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(file_path):
                async with sem:
                    return await process_document(file_path, args.output, args.file_format)
            
            # 创建任务并显示进度条
            tasks = [process_with_semaphore(file_path) for file_path in file_paths]
            results = await tqdm_async.gather(*tasks, desc="文档处理进度")
            
            # 统计处理结果
            successful = sum(1 for r in results if r is not None)
            logger.info(f"处理完成: 成功 {successful}/{len(file_paths)} 个文件")
            
            # 可选: 生成汇总报告
            summary_path = Path(args.output) / "processing_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                summary = {
                    "total_files": len(file_paths),
                    "successful": successful,
                    "failed": len(file_paths) - successful,
                    "settings": {
                        "llm_concurrency": args.concurrency,
                        "file_concurrency": max_concurrent,
                        "output_format": args.format,
                        "file_format": args.file_format
                    }
                }
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"处理汇总已保存到: {summary_path}")
        else:
            logger.warning(f"未找到可处理的文件: {args.input}")
    else:
        logger.error(f"无效的输入路径: {args.input}")

if __name__ == "__main__":
    # 设置更大的默认并发数
    if not hasattr(config, 'MAX_LLM_CONCURRENCY') or config.MAX_LLM_CONCURRENCY < 10:
        config.MAX_LLM_CONCURRENCY = 10
    
    # 增加异步事件循环的最大任务数
    try:
        asyncio.get_event_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=20)
        )
    except RuntimeError:
        # 如果事件循环已关闭，创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=20)
        )
    
    # 设置更大的打开文件限制
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
        logger.debug(f"设置文件描述符限制: {min(hard, 4096)}")
    except (ImportError, ValueError) as e:
        logger.debug(f"无法设置文件描述符限制: {str(e)}")  # Windows或其他不支持的平台
    
    # 设置更大的 asyncio 任务数限制
    if hasattr(asyncio, 'events'):
        asyncio.events._MAX_COMPLETED_QUEUE_SIZE = 10000
    
    asyncio.run(main()) 