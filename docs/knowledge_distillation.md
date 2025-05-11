# 知识蒸馏指南

FastDatasets提供强大的知识蒸馏功能，帮助您从大型语言模型（LLM）中提取知识，创建高质量的训练数据集。本指南将介绍如何使用知识蒸馏功能。

## 什么是知识蒸馏

在LLM训练领域，知识蒸馏是指从大型预训练模型中提取知识并将其转移到较小模型中的过程。在FastDatasets中，我们实现了**数据级知识蒸馏**，即：

1. 从高质量的问题或指令中提取有价值的模式
2. 使用强大的LLM（如GPT-4等）生成高质量回答
3. 将这些问答对整理成标准格式，用于微调其他模型

这种方法有助于创建更高质量、更多样化的训练数据集，从而改善下游模型的性能。

## 主要功能

知识蒸馏模块提供两种主要功能：

1. **数据集蒸馏（Distill）**：从Huggingface等来源采样数据，并使用LLM生成高质量回答
2. **样本扩增（Augment）**：扩充现有高质量样本，生成语义相似但表达不同的变体

## 使用方法

### 1. 从Huggingface数据集蒸馏

```bash
python scripts/distill_dataset.py --mode distill \
                               --dataset_name open-r1/s1K-1.1 \
                               --sample_size 100 \
                               --formats alpaca,sharegpt
```

参数说明：
- `--mode distill`：指定使用蒸馏模式
- `--dataset_name`：Huggingface数据集名称
- `--sample_size`：采样数量（默认为配置文件中的值）
- `--formats`：输出格式，支持alpaca和sharegpt（多个格式以逗号分隔）

### 2. 扩增高质量样本

```bash
python scripts/distill_dataset.py --mode augment \
                               --high_quality_file data/high_quality_samples.json \
                               --num_aug 3
```

参数说明：
- `--mode augment`：指定使用扩增模式
- `--high_quality_file`：包含高质量样本的JSON文件
- `--num_aug`：每个样本生成的变体数量

### 3. 其他常用参数

- `--instruction_col`：指令字段名（如不存在将自动适配）
- `--input_col`：输入字段名（如不存在将自动适配）
- `--output_col`：输出字段名（如不存在将自动适配）
- `--include_reasoning`：是否将推理内容包含在输出中
- `--max_output_tokens`：LLM输出的最大token数
- `--skip_generation`：是否跳过输出生成阶段（保留数据集原始输出）

## 输出格式

知识蒸馏过程会在output目录中创建带时间戳的子目录，并生成以下文件：

- `distilled.json`：蒸馏后的原始数据
- `distilled-alpaca.json`：Alpaca格式的蒸馏数据（指令/输入/输出）
- `distilled-sharegpt.json`：ShareGPT格式的蒸馏数据（对话形式）

## 最佳实践

1. **数据集选择**：选择与您目标任务相关的高质量数据集进行蒸馏
2. **采样策略**：根据数据集大小和多样性调整采样数量
3. **提示词优化**：根据需要调整`app/core/prompt.py`中的提示词模板
4. **质量控制**：使用`--skip_generation`选项查看原始数据，评估是否需要进一步处理
5. **迭代优化**：生成数据集后，审查质量并迭代改进过程

## 高级配置

在`.env`文件中，可以设置以下环境变量来配置知识蒸馏过程：

```
# LLM配置
LLM_API_KEY=your-api-key
LLM_API_BASE=https://api.example.com/v1
LLM_MODEL=gpt-4

# 输出格式
OUTPUT_FORMATS=alpaca,sharegpt

# 默认采样数量
DEFAULT_SAMPLE_SIZE=100

# LLM并发限制
MAX_LLM_CONCURRENCY=5
```

## 故障排除

1. **处理速度慢**：增加MAX_LLM_CONCURRENCY值或减少采样数量
2. **LLM调用错误**：检查API密钥和基础URL是否正确配置
3. **内存不足**：减小采样数量或分批处理数据
4. **字段映射错误**：使用`--instruction_col`等参数手动指定字段名

## 数据集字段映射

FastDatasets会自动尝试识别数据集中的字段，支持以下字段名称：

- 指令字段：instruction, input, prompt, query, question
- 输入字段：input_text, context, source
- 输出字段：content, output, response, answer, completion, target
- 推理字段：reasoning_content, reasoning, rationale, explanation, thinking

## 特殊数据集处理

目前支持特殊处理的数据集：
- Congliu/Chinese-DeepSeek-R1-Distill-data-110k
- open-r1/SYNTHETIC-1-SFT-Data-Code_decontaminated

## 输出格式

#### Alpaca格式
```json
{
    "instruction": "指令内容",
    "input": "输入内容",
    "output": "输出内容"
}
```

#### ShareGPT格式
```json
{
    "conversations": [
        {
            "from": "human",
            "value": "指令: 指令内容\n\n输入: 输入内容"
        },
        {
            "from": "assistant",
            "value": "输出内容"
        }
    ]
}
```

## 常见问题

1. 字段识别失败
   - 检查数据集字段名称
   - 使用`--instruction_col`等参数手动指定字段

2. 输出质量不佳
   - 检查大模型配置
   - 调整提示词模板
   - 增加输出token数

3. 内存不足
   - 减小采样数量
   - 分批处理数据

## 进阶用法

### 自定义提示词模板

可以通过修改`app/core/prompt.py`中的提示词模板来自定义生成过程：

```python
@staticmethod
def generate_output(instruction: str, input_text: str = "") -> str:
    prompt = f"指令: {instruction}\n\n"
    if input_text:
        prompt += f"输入: {input_text}\n\n"
    prompt += "请根据以上指令和输入生成合适的输出。"
    return prompt
```

### 环境变量配置

在`.env`文件中配置相关参数：

```bash
# 数据集配置
DEFAULT_SAMPLE_SIZE=3
OUTPUT_FORMATS=alpaca,sharegpt

# LLM配置
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
LLM_API_BASE=your-api-base
``` 