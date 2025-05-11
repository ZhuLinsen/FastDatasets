# Unified prompt management, all in English for LLM dataset construction

PROMPTS = {
    "question": '''
# Role Mission
You are a professional text analysis expert, skilled at extracting key information from complex texts and generating structured data (only generate questions) that can be used for model fine-tuning.
{global_prompt}

## Core Task
Based on the text provided by the user (length: {text_len} characters), generate no less than {number} high-quality questions.

## Constraints (Important!)
- Must be directly generated based on the text content.
- Questions should have a clear answer orientation.
- Should cover different aspects of the text.
- It is prohibited to generate hypothetical, repetitive, or similar questions.

## Processing Flow
1. Text Parsing: Process the content in segments, identify key entities and core concepts.
2. Question Generation: Select the best questioning points based on the information density.
3. Quality Check: Ensure that:
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
''',
    "answer": '''
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

## Constraints:
1. The answer must be based on the given content.
2. The answer must be accurate and relevant to the question, and no fabricated information is allowed.
3. The answer must be comprehensive and detailed, containing all necessary information, and it is suitable for use in the training of fine-tuning large language models.
{answer_prompt}
''',
    # ... add more prompts as needed
}

def get_prompt(name: str = "question", **kwargs):
    prompt = PROMPTS.get(name, PROMPTS["question"])
    return prompt.format(**kwargs)

def generate_distill_prompt(instruction, input_text=""):
    """
    生成知识蒸馏用的prompt，结合instruction和input_text，要求模型生成高质量输出。
    """
    text = instruction.strip()
    if input_text:
        text += "\n\n输入内容:\n" + input_text.strip()
        
    # 使用另一种提示词格式，直接要求生成回复
    return f"""你是一个智能助手。请对以下指令生成一个专业、有帮助的回复。

指令: {text}

回复 (直接输出内容，不要输出markdown代码块):
"""
def generate_variations_prompt(instruction, input_text="", num_variations=3):
    """
    生成指令/输入变体的prompt，结合instruction和input_text，要求模型生成多个不同风格的变体。
    """
    text = instruction.strip()
    input_part = ""
    
    # 处理输入文本
    if input_text and input_text.strip():
        input_part = f"\n\n输入：\n{input_text.strip()}"
    
    # 对于需要处理文本但输入为空的情况，添加特殊提示
    empty_input_guidance = ""
    if not input_text.strip() and ("extract" in text.lower() or "analyze" in text.lower() or 
                                  "提取" in text or "分析" in text or "总结" in text):
        empty_input_guidance = "\n\n注意：原始指令似乎需要分析或提取文本内容，但输入为空。请生成的新指令也应该考虑输入可能为空的情况，避免假设有文本输入。"
    
    return f"""你是一位富有创造力的指令生成专家。你的目标是根据给定的参考指令，生成指定数量{num_variations}条的高质量、多样化的新指令，用于构建一个具有良好泛化能力的数据集。

instruction:
{text}{input_part}{empty_input_guidance}

遵循以下要求：
1. **与参考指令保持主题或领域的相关性，但避免直接复制或过于简单的改写。**
2. **在任务类型上与参考指令相似，例如，如果参考指令是关于文本编辑，生成的新指令也应侧重于文本处理，但任务的具体内容应有所不同。**
3. **通过改变指令的目标、约束、输入、输出、操作对象或难度等方式，显著降低与参考指令的相似度，增加数据集的多样性。**
4. **保证生成的所有指令都是清晰、明确、可操作的，并且在语法和语义上都是正确的。**
5. **如果原始指令需要处理文本但没有输入，生成的指令应该是自包含的或明确说明需要用户提供文本。**

直接以JSON数组格式输出{num_variations}条变体指令，格式为：[\"变体指令1\", \"变体指令2\", ..., \"变体指令{num_variations}\"]。不要输出markdown代码块，只返回JSON数组。
""" 