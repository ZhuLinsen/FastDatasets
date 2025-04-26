import os
import re
from pathlib import Path
from typing import List, Dict, Any
import textract  # 导入 textract 库
from app.core.logger import logger
from app.core.config import config

class DocumentProcessor:
    """文档处理器，用于解析和处理文档内容"""
    
    def __init__(self):
        self.supported_formats = [".pdf", ".docx", ".txt", ".md"]
        self.chunk_min_len = config.CHUNK_MIN_LEN
        self.chunk_max_len = config.CHUNK_MAX_LEN
        logger.info("DocumentProcessor 初始化，使用 textract 解析文档")
    
    def parse_document(self, file_path: str) -> str:
        """
        解析文档内容
        
        Args:
            file_path: 文档路径
            
        Returns:
            str: 文档内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return ""
            
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_formats:
            logger.error(f"不支持的文件格式: {suffix}")
            return ""
            
        logger.info(f"解析文档: {file_path}")
        
        try:
            # 使用 textract 解析二进制文档 (.pdf, .docx 等)
            if suffix in [".pdf", ".docx"]:
                return self._parse_with_textract(file_path)
            # 使用内置方法解析文本文档 (.txt, .md)
            elif suffix in [".txt", ".md"]:
                return self._parse_txt(file_path)
            
            return ""
        except Exception as e:
            logger.error(f"解析文档失败 {file_path}: {str(e)}")
            return ""
    
    def process_document(self, file_path: str, chunk_size: int = None, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        处理文档内容，分割成多个块
        
        Args:
            file_path: 文档路径
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            List[Dict[str, Any]]: 文档块列表
        """
        content = self.parse_document(file_path)
        if not content:
            return []
            
        if chunk_size is None:
            chunk_size = self.chunk_max_len
        
        # 处理 Markdown 文档
        if file_path.lower().endswith('.md'):
            return self._process_markdown(file_path, content)
            
        # 分割文档
        chunks = self._split_text(content, chunk_size, chunk_overlap)
        
        # 创建文档块列表
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "id": f"{Path(file_path).stem}_{i}",
                "content": chunk,
                "summary": self._generate_summary(chunk),
                "file": str(file_path),
                "chunk_id": f"{Path(file_path).stem}_part_{i+1}",
                "metadata": {
                    "source": str(file_path),
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
            
        logger.info(f"文档已分割为 {len(result)} 个块")
        return result
    
    def _parse_txt(self, file_path: Path) -> str:
        """解析 TXT 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果 UTF-8 失败，尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"解析 TXT 文件失败 (GBK): {str(e)}")
                return ""
        except Exception as e:
            logger.error(f"解析 TXT 文件失败: {str(e)}")
            return ""
    
    def _parse_with_textract(self, file_path: Path) -> str:
        """使用 textract 解析文档"""
        try:
            logger.info(f"使用 textract 解析文件: {file_path}")
            text = textract.process(str(file_path)).decode('utf-8')
            return text
        except UnicodeDecodeError:
            try:
                # 尝试其他编码
                text = textract.process(str(file_path)).decode('gbk')
                return text
            except Exception as e:
                logger.error(f"textract 解析文件失败 (GBK): {str(e)}")
                # 如果解析失败，返回占位内容
                return f"无法解析文档 {file_path.name}，这是一个占位内容。"
        except Exception as e:
            logger.error(f"textract 解析文件失败: {str(e)}")
            # 如果解析失败，返回占位内容
            return f"无法解析文档 {file_path.name}，这是一个占位内容。"
    
    def _extract_outline(self, text: str) -> List[Dict[str, Any]]:
        """提取 Markdown 文本大纲（所有标题）"""
        outline = []
        for match in re.finditer(r'^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$', text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            outline.append({'level': level, 'title': title, 'position': match.start()})
        return outline
    
    def _split_by_headings(self, text: str, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按标题分割 Markdown 文本"""
        if not outline:
            return [{'heading': None, 'level': 0, 'content': text, 'position': 0}]
        sections = []
        # 前言
        if outline[0]['position'] > 0:
            front = text[:outline[0]['position']].strip()
            if front:
                sections.append({'heading': None, 'level': 0, 'content': front, 'position': 0})
        # 标题分块
        for i, current in enumerate(outline):
            next_pos = outline[i+1]['position'] if i+1 < len(outline) else len(text)
            heading_line = text[current['position']:].split('\n', 1)[0]
            start = current['position'] + len(heading_line) + 1
            content = text[start:next_pos].strip()
            sections.append({
                'heading': current['title'],
                'level': current['level'],
                'content': content,
                'position': current['position']
            })
        return sections
    
    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """为文本块生成摘要"""
        if not text:
            return ""
        # 简单实现：取前N个字符
        return text[:max_length].replace("\n", " ")
    
    def _process_markdown(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """处理 Markdown 内容"""
        outline = self._extract_outline(content)
        sections = self._split_by_headings(content, outline)
        chunks = self._split_sections(sections, outline)
        
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "id": f"{file_path.stem}_{i}",
                "content": chunk['content'],
                "summary": chunk['summary'],
                "file": str(file_path),
                "chunk_id": f"{file_path.stem}_part_{i+1}",
                "metadata": {
                    "source": str(file_path),
                    "chunk_id": i,
                    "heading": chunk.get('heading', ''),
                    "total_chunks": len(chunks)
                }
            })
        
        return result
    
    def _generate_section_summary(self, section: Dict[str, Any], outline: List[Dict[str, Any]]) -> str:
        """为 Markdown 段落生成摘要"""
        content = section.get('content', '').strip()
        if content:
            return content[:100]
        if (not section.get('heading') and section.get('level', 0) == 0):
            doc_title = outline[0]['title'] if outline and outline[0]['level'] == 1 else '文档'
            return f"{doc_title} 前言"
        if section.get('heading'):
            idx = next((i for i, o in enumerate(outline) if o['title'] == section['heading'] and o['level'] == section['level']), -1)
            if idx == -1:
                return section['heading']
            parent_titles = []
            parent_level = section['level'] - 1
            for i in range(idx-1, -1, -1):
                if outline[i]['level'] == parent_level:
                    parent_titles.insert(0, outline[i]['title'])
                    parent_level -= 1
            if parent_titles:
                return ' > '.join(parent_titles) + ' > ' + section['heading']
            return section['heading']
        return '未命名段落'
    
    def _split_sections(self, sections: List[Dict[str, Any]], outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并过短段落，拆分过长段落，生成摘要"""
        result = []
        buffer = None
        for section in sections:
            content = section['content'].strip()
            if len(content) < self.chunk_min_len:
                if buffer:
                    buffer['content'] += '\n\n' + (f"{'#'*section['level']} {section['heading']}\n" if section['heading'] else '') + content
                else:
                    buffer = section.copy()
            else:
                if buffer:
                    merged = buffer['content'] + '\n\n' + content
                    if len(merged) <= self.chunk_max_len:
                        buffer['content'] = merged
                        result.append({
                            'summary': self._generate_section_summary(buffer, outline), 
                            'content': buffer['content'],
                            'heading': buffer.get('heading')
                        })
                        buffer = None
                    else:
                        result.append({
                            'summary': self._generate_section_summary(buffer, outline), 
                            'content': buffer['content'],
                            'heading': buffer.get('heading')
                        })
                        result.append({
                            'summary': self._generate_section_summary(section, outline), 
                            'content': content,
                            'heading': section.get('heading')
                        })
                        buffer = None
                else:
                    if len(content) > self.chunk_max_len:
                        # 按句子或定长切分
                        sentences = re.split(r'(?<=[。！？.!?])', content)
                        chunk = ''
                        for sent in sentences:
                            if len(chunk) + len(sent) > self.chunk_max_len:
                                result.append({
                                    'summary': self._generate_section_summary(section, outline), 
                                    'content': chunk,
                                    'heading': section.get('heading')
                                })
                                chunk = sent
                            else:
                                chunk += sent
                        if chunk:
                            result.append({
                                'summary': self._generate_section_summary(section, outline), 
                                'content': chunk,
                                'heading': section.get('heading')
                            })
                    else:
                        result.append({
                            'summary': self._generate_section_summary(section, outline), 
                            'content': content,
                            'heading': section.get('heading')
                        })
        if buffer:
            result.append({
                'summary': self._generate_section_summary(buffer, outline), 
                'content': buffer['content'],
                'heading': buffer.get('heading')
            })
        return result
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            List[str]: 文本块列表
        """
        if not text:
            return []
            
        # 尝试按句子分割
        sentences = re.split(r'(?<=[。！？.!?])', text)
        
        # 组合句子成块
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # 如果当前块 + 新句子超过最大长度，保存当前块并开始新块
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # 保留部分重叠内容
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # 找到最后一个句子结束符的位置
                    overlap_pos = max(current_chunk.rfind("。"), current_chunk.rfind("！"), 
                                   current_chunk.rfind("？"), current_chunk.rfind("."), 
                                   current_chunk.rfind("!"), current_chunk.rfind("?"))
                    
                    if overlap_pos != -1 and overlap_pos >= len(current_chunk) - chunk_overlap:
                        current_chunk = current_chunk[overlap_pos+1:]
                    else:
                        current_chunk = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                else:
                    current_chunk = ""
            
            current_chunk += sentence
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        # 如果没有成功分块，回退到简单的字符分割
        if not chunks:
            chunks = []
            i = 0
            text_len = len(text)
            
            while i < text_len:
                end = min(i + chunk_size, text_len)
                chunk = text[i:end]
                chunks.append(chunk)
                i += chunk_size - chunk_overlap
                
        return chunks 