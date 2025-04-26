# ================== 配置参数 ==================
# chunk长度范围
CHUNK_MIN_LEN = 1500
CHUNK_MAX_LEN = 2000
# OpenAI API配置
API_KEY = "sk-xxx"  # 替换为你的API Key
BASE_URL = "http://10.10.215.251:38888/v1"
# MODEL_NAME = "Qwen2.5-32B-Instruct-GPTQ-Int4"
BASE_URL = "http://10.10.215.250:12345/v1"
MODEL_NAME = 'deepseek-v3'
API_KEY = 'lvlei'
# 输入输出路径
INPUT_PATH = "/home/linsen.zhu/RAG/LightRAG/feishu/files.bak"
OUTPUT_DIR = "fast_datasets/output/"
# 语言（可选：'中文' 或 'English'）
LANGUAGE = '中文'
SYSTEM_PROMPT = "你是Lucky, 由不鸣科技(BoomingTech)构建。你是一个非常有用的助手"
OUTPUT_FORMATS = ["alpaca", "sharegpt"]  # 可多选
OUTPUT_FILE_FORMAT = "json"  # 可选: 'json', 'jsonl'
ENABLE_COT = False  # 是否生成思维链
ENABLE_LABEL = False  # 是否生成标签
ENABLE_OPTIMIZE = True  # 是否优化答案/思维链/标题
MAX_LLM_CONCURRENCY = 100 

# ================== Prompt模板 ==================
QUESTION_PROMPT_CN = '''
# 角色使命
你是一位专业的文本分析专家，擅长从复杂文本中提取关键信息并生成可用于模型微调的结构化数据（仅生成问题）。
{global_prompt}

## 核心任务
根据用户提供的文本（长度：{text_len} 字），生成不少于 {number} 个高质量问题。

## 约束条件（重要！）
- 必须基于文本内容直接生成
- 问题应具有明确答案指向性
- 需覆盖文本的不同方面
- 禁止生成假设性、重复或相似问题

## 处理流程
1. 【文本解析】分段处理内容，识别关键实体和核心概念
2. 【问题生成】基于信息密度选择最佳提问点
3. 【质量检查】确保：
   - 问题答案可在原文中找到依据
   - 标签与问题内容强相关
   - 无格式错误

## 输出格式
- JSON 数组格式必须正确
- 字段名使用英文双引号
- 输出的 JSON 数组必须严格符合以下结构：
```json
["问题1", "问题2", "..."]
```

## 输出示例
```json
[ "人工智能伦理框架应包含哪些核心要素？","民法典对个人数据保护有哪些新规定？"]
```

## 待处理文本
{text}

## 限制
- 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
- 生成不少于{number}个高质量问题
- 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
- 问题不得包含【报告、文章、文献、表格】中提到的这种话术，必须是一个自然的问题
{question_prompt}
'''

QUESTION_PROMPT_EN = '''
# Role Mission
You are a professional text analysis expert, skilled at extracting key information from complex texts and generating structured data(only generate questions) that can be used for model fine-tuning.
{global_prompt}

## Core Task
Based on the text provided by the user(length: {text_len} characters), generate no less than {number} high-quality questions.

## Constraints(Important!)
✔️ Must be directly generated based on the text content.
✔️ Questions should have a clear answer orientation.
✔️ Should cover different aspects of the text.
❌ It is prohibited to generate hypothetical, repetitive, or similar questions.

## Processing Flow
1. 【Text Parsing】Process the content in segments, identify key entities and core concepts.
2. 【Question Generation】Select the best questioning points based on the information density.
3. 【Quality Check】Ensure that:
   - The answers to the questions can be found in the original text.
   - The labels are strongly related to the question content.
   - There are no formatting errors.

## Output Format
- The JSON array format must be correct.
- Use English double-quotes for field names.
- The output JSON array must strictly follow the following structure:
```json
["Question 1", "Question 2", "..."]
```

## Output Example
```json
[ "What core elements should an AI ethics framework include?", "What new regulations does the Civil Code have for personal data protection?" ]
```

## Text to be Processed
{text}

## Restrictions
- Must output in the specified JSON format and do not output any other irrelevant content.
- Generate no less than {number} high-quality questions.
- Questions should not be related to the material itself. For example, questions related to the author, chapters, table of contents, etc. are prohibited.
{question_prompt}
'''

ANSWER_PROMPT_CN = '''
# Role: 微调数据集生成专家
## Profile:
- Description: 你是一名微调数据集生成专家，擅长从给定的内容中生成准确的问题答案，确保答案的准确性和相关性，你要直接回答用户问题，所有信息已内化为你的专业知识。
{global_prompt}

## Skills   :
1. 答案必须基于给定的内容
2. 答案必须准确，不能胡编乱造
3. 答案必须与问题相关
4. 答案必须符合逻辑
5. 基于给定参考内容，用自然流畅的语言整合成一个完整答案，不需要提及文献来源或引用标记
   
## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. 首先，分析给定的文件内容
3. 然后，从内容中提取关键信息
4. 接着，生成与问题相关的准确答案
5. 最后，确保答案的准确性和相关性

## 参考内容：
{text}

## 问题
{question}

## Constrains:
1. 答案必须基于给定的内容
2. 答案必须准确，必须与问题相关，不能胡编乱造
3. 答案必须充分、详细、包含所有必要的信息、适合微调大模型训练使用
4. 答案中不得出现 ' 参考 / 依据 / 文献中提到 ' 等任何引用性表述，只需呈现最终结
{answer_prompt}
'''

ANSWER_PROMPT_EN = '''
# Role: Fine-tuning Dataset Generation Expert
## Profile:
- Description: You are an expert in generating fine-tuning datasets, skilled at generating accurate answers to questions from the given content, ensuring the accuracy and relevance of the answers.
{global_prompt}

## Skills:
1. The answer must be based on the given content.
2. The answer must be accurate and not fabricated.
3. The answer must be relevant to the question.
4. The answer must be logical.

## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. First, analyze the given file content.
3. Then, extract key information from the content.
4. Next, generate an accurate answer related to the question.
5. Finally, ensure the accuracy and relevance of the answer.

## Reference Content:
{text}

## Question
{question}

## Constrains:
1. The answer must be based on the given content.
2. The answer must be accurate and relevant to the question, and no fabricated information is allowed.
3. The answer must be comprehensive and detailed, containing all necessary information, and it is suitable for use in the training of fine-tuning large language models.
{answer_prompt}
'''

LABEL_PROMPT_CN = '''
# Role: 领域分类专家 & 知识图谱专家
- Description: 作为一名资深的领域分类专家和知识图谱专家，擅长从文本内容中提取核心主题，构建分类体系，并输出规定 JSON 格式的标签树。
{global_prompt}

## Skills:
1. 精通文本主题分析和关键词提取
2. 擅长构建分层知识体系
3. 熟练掌握领域分类方法论
4. 具备知识图谱构建能力
5. 精通JSON数据结构

## Goals:
1. 分析书籍目录内容
2. 识别核心主题和关键领域
3. 构建两级分类体系
4. 确保分类逻辑合理
5. 生成规范的JSON输出

## Workflow:
1. 仔细阅读完整的书籍目录内容
2. 提取关键主题和核心概念
3. 对主题进行分组和归类
4. 构建一级领域标签
5. 为适当的一级标签添加二级标签
6. 检查分类逻辑的合理性
7. 生成符合格式的JSON输出

    ## 需要分析的目录
    {text}

    ## 限制
1. 一级领域标签数量5-10个
2. 二级领域标签数量1-10个
3. 最多两层分类层级
4. 分类必须与原始目录内容相关
5. 输出必须符合指定 JSON 格式，不要输出 JSON 外其他任何不相关内容
6. 标签的名字最多不要超过 6 个字
7. 在每个标签前加入序号（序号不计入字数）
8. 如果内容无法归类，请输出["其他"]。
{domainTreePrompt}

## OutputFormat:
```json
[
  {{
    "label": "1 一级领域标签",
    "child": [
      {{"label": "1.1 二级领域标签1"}},
      {{"label": "1.2 二级领域标签2"}}
    ]
  }},
  {{
    "label": "2 一级领域标签(无子标签)"
  }}
]
```
'''

LABEL_PROMPT_EN = '''
# Role: Domain Classification Expert & Knowledge Graph Expert
- Description: As a senior domain classification expert and knowledge graph expert, you are skilled at extracting core themes from text content, constructing classification systems, and performing knowledge categorization and labeling.
{global_prompt}

## Skills:
1. Proficient in text theme analysis and keyword extraction.
2. Good at constructing hierarchical knowledge systems.
3. Skilled in domain classification methodologies.
4. Capable of building knowledge graphs.
5. Proficient in JSON data structures.

## Goals:
1. Analyze the content of the book catalog.
2. Identify core themes and key domains.
3. Construct a two - level classification system.
4. Ensure the classification logic is reasonable.
5. Generate a standardized JSON output.

## Workflow:
1. Carefully read the entire content of the book catalog.
2. Extract key themes and core concepts.
3. Group and categorize the themes.
4. Construct primary domain labels (ensure no more than 10).
5. Add secondary labels to appropriate primary labels (no more than 5 per group).
6. Check the rationality of the classification logic.
7. Generate a JSON output that conforms to the format.

    ## Catalog to be analyzed
    {text}

    ## Constraints
1. The number of primary domain labels should be between 5 and 10.
2. The number of secondary domain labels ≤ 5 per primary label.
3. There should be at most two classification levels.
4. The classification must be relevant to the original catalog content.
5. The output must conform to the specified JSON format.
6. The names of the labels should not exceed 6 characters.
7. Do not output any content other than the JSON.
8. Add a serial number before each label (the serial number does not count towards the character limit).
9. If the content cannot be classified, output ["Other"].
{domainTreePrompt}

## OutputFormat:
```json
[
  {{
    "label": "1 Primary Domain Label",
    "child": [
      {{"label": "1.1 Secondary Domain Label 1"}},
      {{"label": "1.2 Secondary Domain Label 2"}}
    ]
  }},
  {{
    "label": "2 Primary Domain Label (No Sub - labels)"
  }}
]
```
'''

ADDLABEL_PROMPT_CN = '''
# Role: 标签匹配专家
- Description: 你是一名标签匹配专家，擅长根据给定的标签数组和问题数组，将问题打上最合适的领域标签。你熟悉标签的层级结构，并能根据问题的内容优先匹配二级标签，若无法匹配则匹配一级标签，最后打上"其他"标签。

## 标签数组：
{label}

## 问题数组：
{question}

## Workflow:
1. Take a deep breath and work on this problem step-by-step.
2. 首先，读取标签数组和问题数组。
3. 然后，遍历问题数组中的每个问题，根据问题的内容匹配标签数组中的标签。
4. 优先匹配二级标签，若无法匹配则匹配一级标签，最后打上"其他"标签。
5. 将匹配到的标签添加到问题对象中，确保不改变原有数据结构。
6. 最后，输出结果数组，确保格式符合要求。

## Constrains:
1. 只新增一个 label 字段，不改变其他任何格式和数据。
2. 必须按照规定格式返回结果。
3. 优先匹配二级标签，若无法匹配则匹配一级标签，最后打上"其他"标签。
4. 确保标签匹配的准确性和一致性。
5. 匹配的标签必须在标签数组中存在，如果不存在，就打上 其他 
6. 如果标签数组为空或无法匹配，请输出"其他"。
7. 输出结果必须是一个数组，每个元素包含 question、label 字段（只输出这个，不要输出任何其他无关内容）

## Output Example:
```json
[
  {{
    "question": "XSS为什么会在2003年后引起人们更多关注并被OWASP列为威胁榜首？",
    "label": "2.2 XSS攻击"
  }}
]
```
'''

ADDLABEL_PROMPT_EN = '''
# Role: Label Matching Expert
- Description: You are a label matching expert, proficient in assigning the most appropriate domain labels to questions based on the given label array and question array.You are familiar with the hierarchical structure of labels and can prioritize matching secondary labels according to the content of the questions.If a secondary label cannot be matched, you will match a primary label.Finally, if no match is found, you will assign the "Other" label.

## Label Array:
{label}

## Question Array:
{question}

## Workflow:
1. Take a deep breath and work on this problem step - by - step.
2. First, read the label array and the question array.
3. Then, iterate through each question in the question array and match the labels in the label array according to the content of the question.
4. Prioritize matching secondary labels.If no secondary label can be matched, match a primary label.Finally, assign the "Other" label.
5. Add the matched label to the question object without changing the original data structure.
6. Finally, output the result array, ensuring that the format meets the requirements.

## Constrains:
1. Only add one "label" field without changing any other format or data.
2. Must return the result in the specified format.
3. Prioritize matching secondary labels.If no secondary label can be matched, match a primary label.Finally, assign the "Other" label.
4. Ensure the accuracy and consistency of label matching.
5. The matched label must exist in the label array.If it does not exist, assign the "Other" label.
6. If the label array is empty or no match is found, output "Other".
7. The output result must be an array, and each element contains the "question" and "label" fields(only output this, do not output any other irrelevant content).

## Output Example:
```json
[
  {{
    "question": "XSS为什么会在2003年后引起人们更多关注并被OWASP列为威胁榜首？",
    "label": "2.2 XSS攻击"
  }}
]
```
'''

COT_PROMPT_CN = '''
# Role: 思维链生成专家
- Description: 你是一名思维链生成专家，擅长为问题生成详细的推理过程。

## 问题：
{question}

## Workflow:
1. 分析问题，分解为多个推理步骤。
2. 详细描述每一步推理过程。
3. 输出完整的思维链。

## Output Example:
<think>首先...然后...最后...</think>
'''

COT_PROMPT_EN = '''
# Role: Chain-of-Thought Generation Expert
- Description: You are an expert in generating detailed reasoning chains for questions.

## Question:
{question}

## Workflow:
1. Analyze the question and break it down into multiple reasoning steps.
2. Describe each reasoning step in detail.
3. Output the complete chain of thought.

## Output Example:
<think>First... Then... Finally...</think>
'''

def clean_optimized_output(text):
    # 去除常见冗余前缀
    text = re.sub(r"^#+\s*优化后的答案内容[:：]?\s*", "", text)
    text = re.sub(r"^优化后的答案内容[:：]?\s*", "", text)
    text = re.sub(r"^#+\s*Optimized answer content[:：]?\s*", "", text)
    text = re.sub(r"^Optimized answer content[:：]?\s*", "", text)
    text = re.sub(r"^#+\s*优化后的思维链内容[:：]?\s*", "", text)
    text = re.sub(r"^优化后的思维链内容[:：]?\s*", "", text)
    text = re.sub(r"^#+\s*Optimized COT content[:：]?\s*", "", text)
    text = re.sub(r"^Optimized COT content[:：]?\s*", "", text)
    return text.strip()

def clean_markdown_json(text):
    # 去除开头和结尾的```、```json、首尾空行
    text = re.sub(r"^\s*```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*```\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    text = re.sub(r"\s*```json\s*$", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # 如果是json数组，尝试解析后再转为字符串
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            # 只保留每个问题字符串，合并为多行
            return "\n".join(str(q).strip() for q in obj)
        if isinstance(obj, str):
            return obj.strip()
    except Exception:
        pass
    return text

OPTIMIZE_ANSWER_PROMPT_CN = '''
# Role: 答案优化专家
- Description: 你是一名答案优化专家，擅长根据建议对答案进行优化。

## 原始答案：
{answer}

## 优化建议：
{advice}

请直接输出优化后的内容，不要包含任何多余的前缀或标题。
'''

OPTIMIZE_ANSWER_PROMPT_EN = '''
# Role: Answer Optimization Expert
- Description: You are an expert in optimizing answers based on suggestions.

## Original Answer:
{answer}

## Advice:
{advice}

Please output only the optimized answer content, without any extra prefix or title.
'''

OPTIMIZE_COT_PROMPT_CN = '''
# Role: 思维链优化专家
- Description: 你是一名思维链优化专家，擅长根据建议对思维链进行优化。

## 原始思维链：
{cot}

## 优化建议：
{advice}

请直接输出优化后的内容，不要包含任何多余的前缀或标题。
'''

OPTIMIZE_COT_PROMPT_EN = '''
# Role: COT Optimization Expert
- Description: You are an expert in optimizing chains of thought based on suggestions.

## Original COT:
{cot}

## Advice:
{advice}

Please output only the optimized COT content, without any extra prefix or title.
'''

OPTIMIZE_TITLE_PROMPT_CN = '''
# Role: 标题优化专家
- Description: 你是一名标题优化专家，擅长根据内容摘要优化标题。

## 原始摘要：
{summary}

## Output Example:
优化后的标题
'''

OPTIMIZE_TITLE_PROMPT_EN = '''
# Role: Title Optimization Expert
- Description: You are an expert in optimizing titles based on content summaries.

## Original Summary:
{summary}

## Output Example:
Optimized title
'''

# ================== 依赖与工具函数 ==================
import os
import re
import json
import logging
import asyncio
import httpx
from tqdm.asyncio import tqdm as tqdm_async
import textract
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ========== 文本结构处理 ==========
def extract_outline(text):
    """
    提取 Markdown 文本大纲（所有标题）
    """
    outline = []
    for match in re.finditer(r'^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$', text, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        outline.append({'level': level, 'title': title, 'position': match.start()})
    return outline

def split_by_headings(text, outline):
    """
    按标题分割 Markdown 文本
    """
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

def generate_summary(section, outline):
    """
    生成摘要：优先用正文前100字，否则用标题路径或前言
    """
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

def split_sections(sections, outline, min_len=CHUNK_MIN_LEN, max_len=CHUNK_MAX_LEN):
    """
    合并过短段落，拆分过长段落，生成摘要
    """
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
                    result.append({'summary': generate_summary(buffer, outline), 'content': buffer['content']})
                    buffer = None
                else:
                    result.append({'summary': generate_summary(buffer, outline), 'content': buffer['content']})
                    result.append({'summary': generate_summary(section, outline), 'content': content})
                    buffer = None
            else:
                if len(content) > max_len:
                    # 按句子或定长切分
                    sentences = re.split(r'(?<=[。！？.!?])', content)
                    chunk = ''
                    for sent in sentences:
                        if len(chunk) + len(sent) > max_len:
                            result.append({'summary': generate_summary(section, outline), 'content': chunk})
                            chunk = sent
                        else:
                            chunk += sent
                    if chunk:
                        result.append({'summary': generate_summary(section, outline), 'content': chunk})
                else:
                    result.append({'summary': generate_summary(section, outline), 'content': content})
    if buffer:
        result.append({'summary': generate_summary(buffer, outline), 'content': buffer['content']})
    return result

# ========== 文件切分类 ==========

class FileSplitter:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def load_files(self):
        exts = ('.md', '.txt', '.pdf', '.docx')
        if os.path.isdir(self.file_path):
            self.files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path)
                          if os.path.isfile(os.path.join(self.file_path, f)) and f.lower().endswith(exts)]
        else:
            if self.file_path.lower().endswith(exts):
                self.files = [self.file_path]
            else:
                self.files = []
        logging.info(f"加载文件数: {len(self.files)}")
        return self.files

    def read_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.md', '.txt']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = textract.process(file_path).decode('utf-8')
        except Exception as e:
            logging.error(f"解析文件失败: {file_path}, {e}")
            text = ''
        return text

    def split(self):
        all_chunks = []
        for file in tqdm_async(self.files, desc="切分文件"):
            text = self.read_file(file)
            if not text.strip():
                continue
            outline = extract_outline(text)
            sections = split_by_headings(text, outline)
            chunks = split_sections(sections, outline)
            for idx, chunk in enumerate(chunks):
                chunk['file'] = os.path.basename(file)
                chunk['chunk_id'] = f"{os.path.splitext(os.path.basename(file))[0]}-part-{idx+1}"
            all_chunks.extend(chunks)
        logging.info(f"总切分块数: {len(all_chunks)}")
        return all_chunks

# ========== 异步 LLM 调用 ==========
class AsyncLLM:
    def __init__(self, model_name, base_url, api_key, language='中文', max_concurrency=MAX_LLM_CONCURRENCY):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.language = language
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    async def call_llm(self, prompt, max_tokens=2048*2):
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=120*10) as client:
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                }
                resp = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=data)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return content.strip()

    async def generate_qa(self, chunk, number=5, global_prompt='', question_prompt=''):
        print(f"[QA] 生成问题: {chunk.get('chunk_id', '')} ...")
        if self.language == '中文':
            prompt = QUESTION_PROMPT_CN.format(
                text=chunk['content'],
                text_len=len(chunk['content']),
                number=number,
                global_prompt=global_prompt,
                question_prompt=question_prompt
            )
        else:
            prompt = QUESTION_PROMPT_EN.format(
                text=chunk['content'],
                text_len=len(chunk['content']),
                number=number,
                global_prompt=global_prompt,
                question_prompt=question_prompt
            )
        content = await self.call_llm(prompt)
        try:
            qa_list = json.loads(content)
            # 返回每个问题单独一条
            return [clean_markdown_json(str(q)) for q in qa_list if str(q).strip()]
        except Exception:
            # fallback: 尝试用换行分割
            lines = [line.strip() for line in clean_markdown_json(content).split('\n') if line.strip()]
            return lines if lines else [clean_markdown_json(content)]

    async def generate_answer(self, item, question):
        print(f"[ANS] 生成答案: {item.get('chunk_id', '')} | 问题: {question[:20]} ...")
        summary = item.get('summary', '')
        content = item.get('content', '')
        if summary and content:
            text = f"【摘要】{summary}\n【正文】{content}"
        elif content:
            text = content
        else:
            text = summary
        if self.language == '中文':
            prompt = ANSWER_PROMPT_CN.format(
                text=text,
                question=question,
                global_prompt='',
                answer_prompt=''
            )
        else:
            prompt = ANSWER_PROMPT_EN.format(
                text=text,
                question=question,
                global_prompt='',
                answer_prompt=''
            )
        content = await self.call_llm(prompt)
        try:
            if content.startswith('['):
                answer_list = json.loads(content)
                answer = answer_list[0] if answer_list else ""
            else:
                answer = content
        except Exception:
            answer = content
        return clean_markdown_json(answer)

    async def generate_label(self, chunk):
        print(f"[LABEL] 生成标签: {chunk.get('chunk_id', '')} ...")
        if self.language == '中文':
            prompt = LABEL_PROMPT_CN.format(text=chunk['summary'], global_prompt="", domainTreePrompt="")
        else:
            prompt = LABEL_PROMPT_EN.format(text=chunk['summary'], global_prompt="", domainTreePrompt="")
        content = await self.call_llm(prompt)
        try:
            label = json.loads(content)
            if not label or not isinstance(label, list):
                label = ["其他"]
        except Exception:
            label = ["其他"]
        return label

    async def generate_addlabel(self, question_item, label):
        print(f"[ADDLABEL] 问题标签补充: {question_item.get('chunk_id', '')} | 问题: {question_item.get('question', '')[:20]} ...")
        if not label or not isinstance(label, list):
            return "其他"
        if self.language == '中文':
            prompt = ADDLABEL_PROMPT_CN.format(label=json.dumps(label, ensure_ascii=False), question=json.dumps([question_item['question']], ensure_ascii=False))
        else:
            prompt = ADDLABEL_PROMPT_EN.format(label=json.dumps(label, ensure_ascii=False), question=json.dumps([question_item['question']], ensure_ascii=False))
        content = await self.call_llm(prompt)
        try:
            label_result = json.loads(content)
            if isinstance(label_result, list) and label_result:
                return label_result[0].get('label', '') or "其他"
            elif isinstance(label_result, dict):
                return label_result.get('label', '') or "其他"
            else:
                return content or "其他"
        except Exception:
            return "其他"

    async def generate_cot(self, question):
        print(f"[COT] 生成思维链: {question[:20]} ...")
        if self.language == '中文':
            prompt = COT_PROMPT_CN.format(question=question)
        else:
            prompt = COT_PROMPT_EN.format(question=question)
        content = await self.call_llm(prompt)
        return content

    async def optimize_answer(self, item, answer):
        print(f"[OPT_ANS] 优化答案: {item.get('chunk_id', '')} | 问题: {item.get('question', '')[:20]} ...")
        if self.language == '中文':
            prompt = OPTIMIZE_ANSWER_PROMPT_CN.format(answer=answer, advice="请使答案更准确、简洁、无引用性表述")
        else:
            prompt = OPTIMIZE_ANSWER_PROMPT_EN.format(answer=answer, advice="Please make the answer more accurate, concise, and without citation expressions.")
        content = await self.call_llm(prompt)
        return clean_optimized_output(content)

    async def optimize_cot(self, item, cot):
        print(f"[OPT_COT] 优化思维链: {item.get('chunk_id', '')} | 问题: {item.get('question', '')[:20]} ...")
        if self.language == '中文':
            prompt = OPTIMIZE_COT_PROMPT_CN.format(cot=cot, advice="请使思维链更自然、无引用性表述")
        else:
            prompt = OPTIMIZE_COT_PROMPT_EN.format(cot=cot, advice="Please make the chain of thought more natural and without citation expressions.")
        content = await self.call_llm(prompt)
        return clean_optimized_output(content)

    async def optimize_title(self, summary):
        print(f"[OPT_TITLE] 优化标题: {summary[:20]} ...")
        if self.language == '中文':
            prompt = OPTIMIZE_TITLE_PROMPT_CN.format(summary=summary)
        else:
            prompt = OPTIMIZE_TITLE_PROMPT_EN.format(summary=summary)
        content = await self.call_llm(prompt)
        return content

# ========== 格式化导出 ==========
def export_alpaca(data):
    return [
        {
            "instruction": clean_markdown_json(item["question"]),
            "input": "",
            "output": clean_optimized_output(clean_markdown_json(item["answer"])),
            "system": SYSTEM_PROMPT or ""
        }
        for item in data
    ]

def export_sharegpt(data):
    result = []
    for item in data:
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": clean_markdown_json(item["question"])})
        messages.append({"role": "assistant", "content": clean_optimized_output(clean_markdown_json(item["answer"]))})
        result.append({"messages": messages})
    return result

def save_output(data, output_path, file_format="json"):
    if file_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ========== 主流程（异步） ==========
async def main():
    print("[STAGE] 文档切分与摘要 ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    splitter = FileSplitter(INPUT_PATH)
    splitter.load_files()
    chunks = splitter.split()
    chunks_path = os.path.join(OUTPUT_DIR, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 切块完成: {chunks_path}")

    llm = AsyncLLM(MODEL_NAME, BASE_URL, API_KEY, language=LANGUAGE, max_concurrency=MAX_LLM_CONCURRENCY)

    if ENABLE_LABEL:
        print("[STAGE] 并发生成标签 ...")
        tasks_label = [llm.generate_label(chunk) for chunk in chunks]
        labels = await tqdm_async.gather(*tasks_label)
        for chunk, label in zip(chunks, labels):
            chunk['label'] = label
        labels_path = os.path.join(OUTPUT_DIR, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in chunk.items() if k in ['chunk_id', 'label']} for chunk in chunks], f, ensure_ascii=False, indent=2)
        print(f"[INFO] 标签已保存: {labels_path}")

    print("[STAGE] 并发生成问题 ...")
    tasks_qa = [llm.generate_qa(chunk, number=max(1, len(chunk['content']) // 240)) for chunk in chunks]
    all_questions = []
    for chunk, questions in zip(chunks, await tqdm_async.gather(*tasks_qa), strict=False):
        for q in questions:
            all_questions.append({
                "chunk_id": chunk['chunk_id'],
                "file": chunk['file'],
                "summary": chunk['summary'],
                "content": chunk['content'],
                "question": q if isinstance(q, str) else q.get("question", "")
            })
    if ENABLE_LABEL:
        print("[STAGE] 并发补充问题标签 ...")
        tasks_addlabel = [llm.generate_addlabel(q_item, chunks[q_idx // max(1, len(all_questions)//len(chunks))].get('label', ["其他"])) for q_idx, q_item in enumerate(all_questions)]
        addlabels = await tqdm_async.gather(*tasks_addlabel)
        for q_item, label in zip(all_questions, addlabels):
            q_item['label'] = label

    questions_path = os.path.join(OUTPUT_DIR, "questions.json")
    with open(questions_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 问题已保存: {questions_path}")

    print("[STAGE] 并发生成答案/思维链 ...")
    tasks_ans = []
    for item in all_questions:
        if ENABLE_COT:
            tasks_ans.append(asyncio.gather(
                llm.generate_answer(item, item["question"]),
                llm.generate_cot(item["question"])
            ))
        else:
            tasks_ans.append(llm.generate_answer(item, item["question"]))
    results = await tqdm_async.gather(*tasks_ans)
    all_answers = []
    for idx, item in enumerate(all_questions):
        if ENABLE_COT:
            answer, cot = results[idx]
            answer_item = {**item, "answer": answer, "cot": cot}
        else:
            answer = results[idx]
            answer_item = {**item, "answer": answer}
        all_answers.append(answer_item)
    answers_path = os.path.join(OUTPUT_DIR, "answers.json")
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 答案已保存: {answers_path}")

    if ENABLE_OPTIMIZE:
        print("[STAGE] 并发优化答案/思维链 ...")
        tasks_opt_ans = [llm.optimize_answer(item, item['answer']) for item in all_answers if 'answer' in item]
        opt_answers = await tqdm_async.gather(*tasks_opt_ans)
        for item, opt_ans in zip(all_answers, opt_answers):
            item['answer'] = opt_ans
        if ENABLE_COT:
            tasks_opt_cot = [llm.optimize_cot(item, item['cot']) for item in all_answers if 'cot' in item]
            opt_cots = await tqdm_async.gather(*tasks_opt_cot)
            for item, opt_cot in zip([i for i in all_answers if 'cot' in i], opt_cots):
                item['cot'] = opt_cot
        # 可选：标题优化
        # tasks_opt_title = [llm.optimize_title(item.get('summary', '')) for item in all_answers]
        # opt_titles = await tqdm_async.gather(*tasks_opt_title)
        # for item, opt_title in zip(all_answers, opt_titles):
        #     item['title'] = opt_title

    optimized_path = os.path.join(OUTPUT_DIR, "optimized.json")
    with open(optimized_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 优化后数据已保存: {optimized_path}")

    print("[STAGE] 导出多种格式 ...")
    for fmt in OUTPUT_FORMATS:
        if fmt == "alpaca":
            export_data = export_alpaca(all_answers)
            out_path = os.path.join(OUTPUT_DIR, "dataset-alpaca." + OUTPUT_FILE_FORMAT)
        elif fmt == "sharegpt":
            export_data = export_sharegpt(all_answers)
            out_path = os.path.join(OUTPUT_DIR, "dataset-sharegpt." + OUTPUT_FILE_FORMAT)
        else:
            continue
        save_output(export_data, out_path, file_format=OUTPUT_FILE_FORMAT)
        print(f"[INFO] 已导出: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
