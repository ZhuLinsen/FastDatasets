import os
import sys
import asyncio
import argparse
from pathlib import Path
from app.core.logger import logger
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetBuilder
from app.core.config import config

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

async def process_document(file_path: str, output_dir: str = None):
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
            
            # 保存原始数据集
            output_path = Path(output_dir) / f"{Path(file_path).stem}_dataset.jsonl"
            builder.save_dataset(dataset, output_path)
            logger.info(f"数据集已保存: {output_path}")
            
            # 导出多种格式（如 alpaca、sharegpt）
            if hasattr(config, 'OUTPUT_FORMATS') and config.OUTPUT_FORMATS:
                builder.export_dataset(
                    dataset, 
                    output_dir, 
                    config.OUTPUT_FORMATS, 
                    config.OUTPUT_FILE_FORMAT
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
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 处理输入路径
    input_path = Path(args.input)
    if input_path.is_file():
        # 处理单个文件
        await process_document(str(input_path), args.output)
    elif input_path.is_dir():
        # 处理目录中的所有文件
        tasks = []
        for file_path in input_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
                tasks.append(process_document(str(file_path), args.output))
        
        if tasks:
            # 设置最大并发数
            max_concurrent = min(config.MAX_LLM_CONCURRENCY, 5)  # 限制文件处理的并发数
            sem = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(file_path, output_dir):
                async with sem:
                    return await process_document(file_path, output_dir)
            
            # 重新构建任务列表
            bounded_tasks = []
            for file_path in [t._args[0] for t in tasks]:  # 获取原始任务的文件路径参数
                bounded_tasks.append(process_with_semaphore(file_path, args.output))
            
            await asyncio.gather(*bounded_tasks)
        else:
            logger.warning(f"未找到可处理的文件: {args.input}")
    else:
        logger.error(f"无效的输入路径: {args.input}")

if __name__ == "__main__":
    asyncio.run(main()) 