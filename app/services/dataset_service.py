import os
import re
import json
import logging
import asyncio
from typing import List, Dict, Any
from app.core.config import config
from app.core.llm import AsyncLLM
from app.core.prompt import get_prompt
from app.core.storage import storage
from app.core.logger import logger

class TextSplitter:
    @staticmethod
    def extract_outline(text: str) -> List[Dict[str, Any]]:
        """Extract markdown headings as outline."""
        outline = []
        for match in re.finditer(r'^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$', text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            outline.append({'level': level, 'title': title, 'position': match.start()})
        return outline

    @staticmethod
    def split_by_headings(text: str, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split markdown text by headings."""
        if not outline:
            return [{'heading': None, 'level': 0, 'content': text, 'position': 0}]
        sections = []
        if outline[0]['position'] > 0:
            front = text[:outline[0]['position']].strip()
            if front:
                sections.append({'heading': None, 'level': 0, 'content': front, 'position': 0})
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

    @staticmethod
    def generate_summary(section: Dict[str, Any], outline: List[Dict[str, Any]]) -> str:
        content = section.get('content', '').strip()
        if content:
            return content[:100]
        if (not section.get('heading') and section.get('level', 0) == 0):
            doc_title = outline[0]['title'] if outline and outline[0]['level'] == 1 else 'Document'
            return f"{doc_title} Introduction"
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
        return 'Unnamed Section'

    @staticmethod
    def split_sections(sections: List[Dict[str, Any]], outline: List[Dict[str, Any]], min_len: int, max_len: int) -> List[Dict[str, Any]]:
        result = []
        buffer = None
        for section in sections:
            content = section['content'].strip()
            if len(content) < min_len:
                if buffer:
                    buffer['content'] += '\n\n' + (f"{'#'*section['level']} {section['heading']}\n" if section['heading'] else '') + content
                else:
                    buffer = section.copy()
            else:
                if buffer:
                    merged = buffer['content'] + '\n\n' + content
                    if len(merged) <= max_len:
                        buffer['content'] = merged
                        result.append({'summary': TextSplitter.generate_summary(buffer, outline), 'content': buffer['content']})
                        buffer = None
                    else:
                        result.append({'summary': TextSplitter.generate_summary(buffer, outline), 'content': buffer['content']})
                        result.append({'summary': TextSplitter.generate_summary(section, outline), 'content': content})
                        buffer = None
                else:
                    if len(content) > max_len:
                        sentences = re.split(r'(?<=[.!?。！？])', content)
                        chunk = ''
                        for sent in sentences:
                            if len(chunk) + len(sent) > max_len:
                                result.append({'summary': TextSplitter.generate_summary(section, outline), 'content': chunk})
                                chunk = sent
                            else:
                                chunk += sent
                        if chunk:
                            result.append({'summary': TextSplitter.generate_summary(section, outline), 'content': chunk})
                    else:
                        result.append({'summary': TextSplitter.generate_summary(section, outline), 'content': content})
        if buffer:
            result.append({'summary': TextSplitter.generate_summary(buffer, outline), 'content': buffer['content']})
        return result

class DatasetService:
    def __init__(self):
        self.llm = AsyncLLM()

    def load_files(self, input_path: str) -> List[str]:
        exts = ('.md', '.txt')
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                     if os.path.isfile(os.path.join(input_path, f)) and f.lower().endswith(exts)]
        else:
            files = [input_path] if input_path.lower().endswith(exts) else []
        logger.info(f"Loaded {len(files)} files from {input_path}")
        return files

    def read_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read file: {file_path}, {e}")
            text = ''
        return text

    def split_documents(self, input_path: str = None) -> List[Dict[str, Any]]:
        input_path = input_path or config.INPUT_PATH
        files = self.load_files(input_path)
        all_chunks = []
        for file in files:
            text = self.read_file(file)
            if not text.strip():
                continue
            outline = TextSplitter.extract_outline(text)
            sections = TextSplitter.split_by_headings(text, outline)
            chunks = TextSplitter.split_sections(sections, outline, config.CHUNK_MIN_LEN, config.CHUNK_MAX_LEN)
            for idx, chunk in enumerate(chunks):
                chunk['file'] = os.path.basename(file)
                chunk['chunk_id'] = f"{os.path.splitext(os.path.basename(file))[0]}-part-{idx+1}"
            all_chunks.extend(chunks)
        logger.info(f"Total chunks: {len(all_chunks)}")
        return all_chunks

    async def generate_questions(self, chunk: Dict[str, Any], number: int = 5, global_prompt: str = '', question_prompt: str = '') -> List[str]:
        prompt = get_prompt(
            name="question",
            text=chunk['content'],
            text_len=len(chunk['content']),
            number=number,
            global_prompt=global_prompt,
            question_prompt=question_prompt
        )
        content = await self.llm.call_llm(prompt)
        try:
            qa_list = json.loads(content)
            return [str(q).strip() for q in qa_list if str(q).strip()]
        except Exception:
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            return lines if lines else [content]

    async def generate_answer(self, chunk: Dict[str, Any], question: str) -> str:
        prompt = get_prompt(
            name="answer",
            text=chunk.get('content', ''),
            question=question,
            global_prompt='',
            answer_prompt=''
        )
        content = await self.llm.call_llm(prompt)
        try:
            if content.startswith('['):
                answer_list = json.loads(content)
                answer = answer_list[0] if answer_list else ""
            else:
                answer = content
        except Exception:
            answer = content
        return answer.strip()

    async def generate_label(self, chunk: Dict[str, Any], global_prompt: str = "", label_prompt: str = "") -> list:
        """Generate label for a chunk using LLM."""
        prompt = f"You are a domain classification expert. Analyze the following summary and output a list of domain labels in JSON array format.\nSummary: {chunk.get('summary', '')}\n{global_prompt}\n{label_prompt}"
        content = await self.llm.call_llm(prompt)
        try:
            label = json.loads(content)
            if not label or not isinstance(label, list):
                label = ["Other"]
        except Exception:
            label = ["Other"]
        return label

    async def optimize_answer(self, answer: str, advice: str = "Please make the answer more accurate, concise, and natural.") -> str:
        """Optimize answer using LLM."""
        prompt = f"You are an answer optimization expert.\nOriginal Answer: {answer}\nAdvice: {advice}\nPlease output only the optimized answer content, without any extra prefix or title."
        content = await self.llm.call_llm(prompt)
        return content.strip()

    def export_data(self, data: List[Dict[str, Any]], output_path: str, file_format: str = "json"):
        storage.export(data, output_path, file_format)

    # Add more methods for label generation, optimization, etc. as needed 