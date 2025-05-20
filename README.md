# FastDatasets 

![FastDatasets](fastdatasets.png)

一个强大的工具，用于为大语言模型（LLM）创建高质量的训练数据集 | [Switch to English](README_en.md)

## 主要功能

### 1. 基于自由文档生成数据集
- **智能文档处理**：支持多种格式文档的智能分割
- **问题生成**：基于文档内容自动生成相关问题
- **答案生成**：使用 LLM 生成高质量答案
- **异步处理**：支持大规模文档的异步处理
- **多种导出格式**：支持多种数据集格式导出（Alpaca、ShareGPT等）
- **直接SFT就绪输出**：生成适用于监督微调的数据集

### 2. 数据蒸馏与优化
- **知识蒸馏**：从大模型中提取知识到训练数据集
- **指令扩增**：自动生成指令变体，扩充训练数据
- **质量优化**：使用 LLM 优化和提升数据质量
- **多格式支持**：支持从多种格式的数据集进行蒸馏


## 快速开始

### 环境要求

- Python 3.8+
- 依赖包：见 `requirements.txt`

### 安装

```bash
# 克隆仓库
git clone https://github.com/ZhuLinsen/FastDatasets.git
cd FastDatasets

# 创建并激活虚拟环境（可选）
conda create -n fast_datasets python=3.10
conda activate fast_datasets

# 安装依赖
pip install -e .

# 创建环境配置文件
cp .env.example .env
# 使用编辑器修改.env文件，配置大模型API等必要信息
```

### 使用示例

1. 首先配置您的 LLM
```bash
# 创建并编辑.env文件，从.env.example复制并修改
cp .env.example .env
# 编辑.env文件并配置您的大模型API
# LLM_API_KEY=your_api_key_here
# LLM_API_BASE=https://api.deepseek.com/v1 (或其他模型API地址)

# 使用以下命令测试LLM连接和能力
python scripts/test_llm.py
```

2. 从文档生成数据集
```bash
# 使用测试文档生成数据集
python scripts/dataset_generator.py tests/test.txt -o ./output

# 处理自定义文档
python scripts/dataset_generator.py 路径/到/你的文档.pdf -o ./输出目录

# 处理整个目录下的文档
python scripts/dataset_generator.py 路径/到/文档目录/ -o ./输出目录
```

3. 使用知识蒸馏创建训练数据集
```bash
# 从Huggingface数据集提取问题并生成高质量回答
python scripts/distill_dataset.py --mode distill --dataset_name open-r1/s1K-1.1 --sample_size 10

# 使用高质量样本生成变体并进行知识蒸馏(可选)
python scripts/distill_dataset.py --mode augment --high_quality_file data/high_quality_samples.json --num_aug 3
```

更多详细用法，请参考：[知识蒸馏指南](docs/knowledge_distillation.md) | [文档处理与数据集生成](docs/custom_data_conversion.md)

### 支持的文档格式

- PDF (*.pdf)
- Word (*.docx)
- Markdown (*.md)
- 纯文本 (*.txt)
- 其他主流格式文档

## 输出结构

### 文档处理输出
处理完成后，在指定的输出目录会生成以下文件：

- `chunks.json`: 文档切分后的文本块
- `questions.json`: 生成的问题
- `answers.json`: 生成的答案
- `optimized.json`: 优化后的问答对
- `dataset-alpaca.json`: Alpaca 格式的数据集
- `dataset-sharegpt.json`: ShareGPT 格式的数据集（如果配置了此输出格式）

### 知识蒸馏输出
知识蒸馏过程会在output目录下创建带时间戳的子目录，并生成以下文件：

- `distilled.json`: 蒸馏后的原始数据
- `distilled-alpaca.json`: Alpaca 格式的蒸馏数据
- `distilled-sharegpt.json`: ShareGPT 格式的蒸馏数据

## 高级用法

### API 服务

启动 API 服务，提供 Web 界面和 RESTful API：

```bash
python -m app.main
```

默认地址：http://localhost:8000

### 故障排除

1. **处理速度慢**：增加 MAX_LLM_CONCURRENCY 值，或减少 LLM_MAX_TOKENS
2. **内存不足**：减小处理的文档数量或文档大小
3. **API 超时**：检查网络连接，或减少 MAX_LLM_CONCURRENCY

## 许可证
[Apache 2.0](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZhuLinsen/FastDatasets&type=Date)](https://www.star-history.com/#ZhuLinsen/FastDatasets&Date)

