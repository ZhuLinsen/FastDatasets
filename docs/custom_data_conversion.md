# 文档处理与数据集生成指南

本文档介绍如何使用FastDatasets从各种文档(如PDF、Word、Markdown等)生成高质量的训练数据集。

## 文档处理流程

FastDatasets文档处理流程包括以下步骤：

1. **文档加载与分块**：将文档加载并分割成适当大小的文本块
2. **问题生成**：基于文本块内容自动生成相关问题
3. **答案生成**：使用LLM根据问题和文本内容生成高质量答案
4. **数据集导出**：将生成的问答对导出为训练数据集格式

## 使用命令行工具

FastDatasets提供了简单易用的命令行工具来处理文档：

```bash
# 处理单个文档
python scripts/dataset_generator.py path/to/your/document.pdf -o ./output_dir

# 处理整个目录下的文档
python scripts/dataset_generator.py path/to/document/directory/ -o ./output_dir

# 指定输出格式
python scripts/dataset_generator.py your_document.pdf -o ./output -f sharegpt
```

参数说明：
- 第一个参数: 文档路径或目录路径
- `-o`, `--output`: 输出目录，默认为"output"
- `-f`, `--format`: 输出格式，支持"alpaca"或"sharegpt"，默认为环境变量中设置的`OUTPUT_FORMAT`

## 支持的文档格式

FastDatasets支持多种文档格式处理：

- PDF (*.pdf)
- Word (*.docx)
- Markdown (*.md)
- 纯文本 (*.txt)
- 其他通过langchain支持的格式

## 输出文件介绍

处理完成后，在指定的输出目录会生成以下文件：

- `chunks.json`: 文档切分后的文本块
- `questions.json`: 基于文本块生成的问题
- `answers.json`: 基于问题和文本块生成的答案
- `optimized.json`: 优化后的问答对（如启用了优化）
- `dataset-alpaca.json`: Alpaca格式的数据集
- `dataset-sharegpt.json`: ShareGPT格式的数据集（如配置了此输出格式）

## 输出格式示例

### Alpaca格式 (默认)

```json
[
  {
    "instruction": "解释这段文本中的主要概念",
    "input": "人工智能是计算机科学的一个分支...",
    "output": "这段文本主要介绍了人工智能的概念..."
  }
]
```

### ShareGPT格式

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "解释这段文本中的主要概念\n\n人工智能是计算机科学的一个分支..."
      },
      {
        "from": "assistant",
        "value": "这段文本主要介绍了人工智能的概念..."
      }
    ]
  }
]
```

## 配置文档处理参数

您可以在`.env`文件中配置文档处理参数：

```
# 文档处理配置
DOCUMENT_MIN_CHUNK_SIZE=1500    # 最小文档块大小 
DOCUMENT_MAX_CHUNK_SIZE=2000    # 最大文档块大小
DOCUMENT_CHUNK_SIZE=1000        # 默认分块大小
DOCUMENT_CHUNK_OVERLAP=200      # 分块重叠大小

# 输出格式
OUTPUT_FORMAT=alpaca            # 输出格式类型：alpaca或sharegpt
```

## 高级用法

### 代码方式调用

除了命令行工具外，您还可以在Python代码中直接使用文档处理功能：

```python
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetGenerator

# 初始化处理器
doc_processor = DocumentProcessor()
dataset_generator = DatasetGenerator()

# 处理文档
chunks = doc_processor.process_document("your_document.pdf")
questions = dataset_generator.generate_questions(chunks)
answers = dataset_generator.generate_answers(chunks, questions)

# 导出数据集
dataset = dataset_generator.create_dataset(questions, answers)
dataset_generator.export_dataset(dataset, "output/my_dataset.json", format="alpaca")
```

## 最佳实践

1. **文档分块调整**：根据文档复杂度调整分块大小，技术文档可能需要更小的分块
2. **质量审查**：生成数据后，建议手动审查部分样本质量
3. **环境配置**：为获得最佳结果，推荐使用性能较好的LLM模型
4. **并发处理**：处理大型文档集时，可增加`MAX_LLM_CONCURRENCY`值提高并行度

## 注意事项

1. 确保配置了有效的LLM API密钥和URL
2. 大型文档处理可能耗时较长，请耐心等待
3. 图表、表格等非文本内容可能无法被正确处理
4. 对于敏感文档，请在处理前确保已获得适当权限 