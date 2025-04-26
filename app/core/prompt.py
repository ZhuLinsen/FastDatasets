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