import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from app.core.logger import logger
from app.core.document import DocumentProcessor

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 加载环境变量
load_dotenv()

def test_pdf_parsing():
    """测试 PDF 文档解析"""
    try:
        # 获取测试文件路径
        test_file = Path("tests/data/test.pdf")
        if not test_file.exists():
            logger.error(f"测试文件不存在: {test_file}")
            return False
            
        # 初始化文档处理器
        processor = DocumentProcessor()
        
        # 解析文档
        logger.info(f"开始解析文档: {test_file}")
        content = processor.parse_document(test_file)
        
        if content:
            logger.info("文档解析成功！")
            logger.info(f"文档内容长度: {len(content)} 字符")
            logger.debug(f"文档内容预览: {content[:200]}...")
            return True
        else:
            logger.error("文档解析失败：未获取到内容")
            return False
            
    except Exception as e:
        logger.error(f"文档解析测试失败: {str(e)}")
        return False

def test_document_processing():
    """测试文档处理流程"""
    try:
        processor = DocumentProcessor()
        test_file = Path("tests/data/test.pdf")
        
        # 处理文档
        logger.info("开始处理文档")
        result = processor.process_document(
            test_file,
            chunk_size=int(os.getenv("DOCUMENT_CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("DOCUMENT_CHUNK_OVERLAP", 200))
        )
        
        if result:
            logger.info("文档处理成功！")
            logger.info(f"生成 {len(result)} 个文本块")
            for i, chunk in enumerate(result[:3], 1):
                logger.debug(f"文本块 {i} 预览: {chunk[:100]}...")
            return True
        else:
            logger.error("文档处理失败")
            return False
            
    except Exception as e:
        logger.error(f"文档处理测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始文档处理测试")
    
    # 测试 PDF 解析
    if not test_pdf_parsing():
        logger.error("PDF 解析测试失败")
        return
        
    # 测试文档处理
    if not test_document_processing():
        logger.error("文档处理测试失败")
        return
        
    logger.info("所有测试完成！文档处理功能工作正常")

if __name__ == "__main__":
    main() 