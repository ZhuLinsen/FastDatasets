import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import asyncio
import httpx
from tqdm.asyncio import tqdm as tqdm_async
from app.core.logger import logger
from app.core.config import config

class DatasetBuilder:
    """数据集构建器，用于从文档块构建训练数据集"""
    
    def __init__(self):
        self.model_name = config.MODEL_NAME
        self.base_url = config.BASE_URL
        self.api_key = config.API_KEY
        self.language = config.LANGUAGE
        self.system_prompt = config.SYSTEM_PROMPT
        self.enable_cot = config.ENABLE_COT
        self.enable_label = config.ENABLE_LABEL
        self.enable_optimize = config.ENABLE_OPTIMIZE
        self.max_concurrency = config.MAX_LLM_CONCURRENCY
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.info("DatasetBuilder 初始化")

    async def build_dataset(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        构建数据集
        
        Args:
            chunks: 文档块列表
            
        Returns:
            List[Dict[str, Any]]: 数据集
        """
        if not chunks:
            logger.warning("没有文档块，无法构建数据集")
            return []
            
        logger.info(f"开始构建数据集，共 {len(chunks)} 个文档块")
        
        # 步骤 1: 为每个文档块生成问题
        all_questions = []
        for chunk in chunks:
            questions = await self._generate_questions(chunk["content"], max(1, len(chunk["content"]) // 240))
            for q in questions:
                all_questions.append({
                    "chunk_id": chunk.get('chunk_id', ''),
                    "file": chunk.get('file', ''),
                    "summary": chunk.get('summary', ''),
                    "content": chunk.get('content', ''),
                    "question": q
                })
        
        # 步骤 2: 为每个问题生成答案
        dataset = []
        for item in all_questions:
            # 生成答案和思维链
            if self.enable_cot:
                answer, cot = await asyncio.gather(
                    self._generate_answer(item["question"], item["content"]),
                    self._generate_cot(item["question"])
                )
                data_point = {**item, "answer": answer, "cot": cot}
            else:
                answer = await self._generate_answer(item["question"], item["content"])
                data_point = {**item, "answer": answer}
            
            # 如果启用标签生成
            if self.enable_label:
                data_point["labels"] = await self._generate_labels(item["question"])
            
            # 如果启用答案优化
            if self.enable_optimize:
                if self.enable_cot:
                    optimized_answer, optimized_cot = await asyncio.gather(
                        self._optimize_answer(data_point["answer"]),
                        self._optimize_cot(data_point["cot"])
                    )
                    data_point["answer"] = optimized_answer
                    data_point["cot"] = optimized_cot
                else:
                    data_point["answer"] = await self._optimize_answer(data_point["answer"])
            
            dataset.append(data_point)
                
        logger.info(f"数据集构建完成，共 {len(dataset)} 个数据点")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str):
        """
        保存数据集
        
        Args:
            dataset: 数据集
            output_path: 输出路径
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 根据文件格式保存
            suffix = Path(output_path).suffix.lower()
            if suffix == ".json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)
            elif suffix == ".jsonl":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                # 默认使用 JSONL 格式
                output_path = str(Path(output_path).with_suffix(".jsonl"))
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        
            logger.info(f"数据集已保存: {output_path}")
        except Exception as e:
            logger.error(f"保存数据集失败: {str(e)}")

    def export_dataset(self, dataset: List[Dict[str, Any]], output_dir: str, formats: List[str], file_format: str = "json"):
        """
        导出数据集为多种格式
        
        Args:
            dataset: 数据集
            output_dir: 输出目录
            formats: 导出格式列表，如 ["alpaca", "sharegpt"]
            file_format: 文件格式，如 "json" 或 "jsonl"
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for fmt in formats:
            if fmt == "alpaca":
                export_data = self._export_alpaca(dataset)
                out_path = os.path.join(output_dir, f"dataset-alpaca.{file_format}")
            elif fmt == "sharegpt":
                export_data = self._export_sharegpt(dataset)
                out_path = os.path.join(output_dir, f"dataset-sharegpt.{file_format}")
            else:
                logger.warning(f"不支持的导出格式: {fmt}")
                continue
                
            self._save_output(export_data, out_path, file_format)
            logger.info(f"已导出 {fmt} 格式: {out_path}")
    
    def _export_alpaca(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """导出为 Alpaca 格式"""
        return [
            {
                "instruction": self._clean_markdown_json(item["question"]),
                "input": "",
                "output": self._clean_optimized_output(self._clean_markdown_json(item["answer"])),
                "system": self.system_prompt or ""
            }
            for item in data
        ]
    
    def _export_sharegpt(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """导出为 ShareGPT 格式"""
        result = []
        for item in data:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": self._clean_markdown_json(item["question"])})
            messages.append({"role": "assistant", "content": self._clean_optimized_output(self._clean_markdown_json(item["answer"]))})
            result.append({"messages": messages})
        return result
    
    def _save_output(self, data: List[Dict[str, Any]], output_path: str, file_format: str = "json"):
        """保存输出文件"""
        if file_format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _clean_markdown_json(self, text: str) -> str:
        """清理 Markdown 中的 JSON 格式"""
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
    
    def _clean_optimized_output(self, text: str) -> str:
        """清理优化后的输出"""
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
    
    async def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """调用 LLM API"""
        async with self.semaphore:
            # 确保从环境变量获取最新的 API 配置
            api_key = os.getenv("API_KEY") or config.API_KEY
            base_url = os.getenv("BASE_URL") or config.BASE_URL
            model_name = os.getenv("MODEL_NAME") or config.MODEL_NAME
            
            # 更新当前实例的设置
            self.api_key = api_key
            self.base_url = base_url
            self.model_name = model_name
            
            # 确保 API URL 格式正确
            if self.base_url and not self.base_url.startswith(('http://', 'https://')):
                self.base_url = f"https://{self.base_url}"
                
            # 检查必要参数
            if not self.api_key or not self.base_url or not self.model_name:
                logger.error("缺少必要的 LLM 配置参数")
                return self._fallback_response(prompt)
            
            retries = 3  # 最大重试次数
            backoff_factor = 1.5  # 退避因子
            
            for attempt in range(retries):
                try:
                    timeout = 60 * (2 if attempt == 0 else attempt * 2)  # 第一次 120 秒，然后逐步增加
                    
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        logger.info(f"调用 LLM API (尝试 {attempt+1}/{retries}): {self.base_url}")
                        
                        data = {
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens
                        }
                        
                        # 确保 Headers 正确
                        headers = {"Authorization": f"Bearer {self.api_key}"}
                        
                        try:
                            resp = await client.post(f"{self.base_url}/chat/completions", 
                                                  headers=headers, 
                                                  json=data, 
                                                  follow_redirects=True)
                            resp.raise_for_status()
                            content = resp.json()["choices"][0]["message"]["content"]
                            return content.strip()
                        except httpx.HTTPStatusError as e:
                            logger.error(f"HTTP 错误: {e.response.status_code} - {e.response.text}")
                            if e.response.status_code == 401:
                                logger.error("API 密钥错误或未授权")
                                # 认证错误不重试
                                break
                            elif e.response.status_code == 429:
                                logger.warning("请求频率限制，将重试")
                            raise
                        
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    if attempt < retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"连接错误: {str(e)}，将在 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"连接失败，已达到最大重试次数: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"调用 LLM API 失败: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    if attempt < retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"将在 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("已达到最大重试次数")
            
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """当 LLM API 调用失败时的后备响应"""
        logger.warning("使用模拟回复代替 LLM 响应")
        if "question" in prompt.lower():
            return json.dumps(["这是一个示例问题？", "这是另一个示例问题？"])
        elif "answer" in prompt.lower():
            return "这是一个示例回答，由于无法连接到 LLM API 而生成的模拟内容。"
        else:
            return "模拟 LLM 响应"

    async def _generate_questions(self, context: str, number: int = 5) -> List[str]:
        """生成问题"""
        logger.info(f"生成问题: 生成 {number} 个问题...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# 角色使命
你是一位专业的文本分析专家，擅长从复杂文本中提取关键信息并生成可用于模型微调的结构化数据（仅生成问题）。

## 核心任务
根据用户提供的文本（长度：{len(context)} 字），生成不少于 {number} 个高质量问题。

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

## 待处理文本
{context}

## 限制
- 必须按照规定的 JSON 格式输出，不要输出任何其他不相关内容
- 生成不少于{number}个高质量问题
- 问题不要和材料本身相关，例如禁止出现作者、章节、目录等相关问题
- 问题不得包含【报告、文章、文献、表格】中提到的这种话术，必须是一个自然的问题
"""
        else:
            prompt = f"""
# Role Mission
You are a professional text analysis expert, skilled at extracting key information from complex texts and generating structured data(only generate questions) that can be used for model fine-tuning.

## Core Task
Based on the text provided by the user(length: {len(context)} characters), generate no less than {number} high-quality questions.

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

## Text to be Processed
{context}

## Restrictions
- Must output in the specified JSON format and do not output any other irrelevant content.
- Generate no less than {number} high-quality questions.
- Questions should not be related to the material itself. For example, questions related to the author, chapters, table of contents, etc. are prohibited.
"""
        
        # 调用 LLM
        content = await self._call_llm(prompt)
        
        try:
            # 尝试解析 JSON
            questions = json.loads(content)
            if isinstance(questions, list):
                return [self._clean_markdown_json(str(q)) for q in questions if str(q).strip()]
            else:
                # 可能返回的是包含问题的对象
                return [self._clean_markdown_json(content)]
        except Exception:
            # Fallback: 尝试用换行分割
            lines = [line.strip() for line in self._clean_markdown_json(content).split('\n') if line.strip()]
            return lines if lines else [self._clean_markdown_json(content)]

    async def _generate_answer(self, question: str, context: str) -> str:
        """生成答案"""
        logger.info(f"生成答案: 问题: {question[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 微调数据集生成专家
## Profile:
- Description: 你是一名微调数据集生成专家，擅长从给定的内容中生成准确的问题答案，确保答案的准确性和相关性，你要直接回答用户问题，所有信息已内化为你的专业知识。

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
{context}

## 问题
{question}

## Constrains:
1. 答案必须基于给定的内容
2. 答案必须准确，必须与问题相关，不能胡编乱造
3. 答案必须充分、详细、包含所有必要的信息、适合微调大模型训练使用
4. 答案中不得出现 ' 参考 / 依据 / 文献中提到 ' 等任何引用性表述，只需呈现最终结
"""
        else:
            prompt = f"""
# Role: Fine-tuning Dataset Generation Expert
## Profile:
- Description: You are an expert in generating fine-tuning datasets, skilled at generating accurate answers to questions from the given content, ensuring the accuracy and relevance of the answers.

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
{context}

## Question
{question}

## Constrains:
1. The answer must be based on the given content.
2. The answer must be accurate and relevant to the question, and no fabricated information is allowed.
3. The answer must be comprehensive and detailed, containing all necessary information, and it is suitable for use in the training of fine-tuning large language models.
"""
        
        # 调用 LLM
        content = await self._call_llm(prompt)
        
        try:
            if content.startswith('['):
                answer_list = json.loads(content)
                answer = answer_list[0] if answer_list else ""
            else:
                answer = content
        except Exception:
            answer = content
        
        return self._clean_markdown_json(answer)

    async def _generate_cot(self, question: str) -> str:
        """生成思维链"""
        logger.info(f"生成思维链: 问题: {question[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
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
"""
        else:
            prompt = f"""
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
"""
        
        # 调用 LLM
        content = await self._call_llm(prompt)
        return content

    async def _generate_labels(self, question: str) -> List[str]:
        """生成标签"""
        logger.info(f"生成标签: 问题: {question[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 领域分类专家
- Description: 你是一名标签分类专家，能够根据问题内容生成相关的标签。

## 问题：
{question}

## 任务：
为该问题生成2-3个相关领域标签，这些标签应该能够概括问题所属的知识领域。

## 输出格式：
[
  "标签1",
  "标签2",
  "标签3"
]
"""
        else:
            prompt = f"""
# Role: Domain Classification Expert
- Description: You are a label classification expert who can generate relevant labels based on the content of questions.

## Question:
{question}

## Task:
Generate 2-3 relevant domain labels for this question. These labels should be able to summarize the knowledge domain to which the question belongs.

## Output Format:
[
  "Label1",
  "Label2",
  "Label3"
]
"""
        
        # 调用 LLM
        content = await self._call_llm(prompt)
        
        try:
            labels = json.loads(content)
            if isinstance(labels, list):
                return labels
            else:
                return ["其他"]
        except Exception:
            return ["其他"]

    async def _optimize_answer(self, answer: str) -> str:
        """优化答案"""
        logger.info(f"优化答案: {answer[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 答案优化专家
- Description: 你是一名答案优化专家，擅长优化答案。

## 原始答案：
{answer}

## 优化建议：
1. 使答案更准确、简洁、无引用性表述
2. 确保答案自然流畅，避免冗余
3. 删除所有引用性表述如"根据文章"、"参考文献表明"等
4. 确保内容与原始答案保持一致，不要添加新信息

请直接输出优化后的内容，不要包含任何多余的前缀或标题。
"""
        else:
            prompt = f"""
# Role: Answer Optimization Expert
- Description: You are an expert in optimizing answers based on suggestions.

## Original Answer:
{answer}

## Optimization Suggestions:
1. Make the answer more accurate, concise, and without citation expressions
2. Ensure the answer is natural and fluent, avoiding redundancy
3. Remove all citation expressions such as "according to the article", "the reference shows", etc.
4. Ensure the content is consistent with the original answer, do not add new information

Please output only the optimized answer content, without any extra prefix or title.
"""
        
        # 调用 LLM
        content = await self._call_llm(prompt)
        return self._clean_optimized_output(content)

    async def _optimize_cot(self, cot: str) -> str:
        """优化思维链"""
        logger.info(f"优化思维链: {cot[:20]}...")
        
        # 构建 prompt
        if self.language == '中文':
            prompt = f"""
# Role: 思维链优化专家
- Description: 你是一名思维链优化专家，擅长优化思维链。

## 原始思维链：
{cot}

## 优化建议：
1. 使思维链更自然、流畅
2. 去除引用性表述
3. 确保逻辑清晰
4. 保持内容与原始思维链一致

请直接输出优化后的内容，不要包含任何多余的前缀或标题。
"""
        else:
            prompt = f"""
# Role: COT Optimization Expert
- Description: You are an expert in optimizing chains of thought.

## Original COT:
{cot}

## Optimization Suggestions:
1. Make the chain of thought more natural and fluent
2. Remove citation expressions
3. Ensure clear logic
4. Maintain consistency with the original chain of thought

Please output only the optimized COT content, without any extra prefix or title.
"""
        
        # 调用 LLM
        content = await self._call_llm(prompt)
        return self._clean_optimized_output(content) 