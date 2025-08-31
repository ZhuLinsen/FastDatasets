# FastDatasets 🚀

![FastDatasets](fastdatasets.png)

[![GitHub Stars](https://img.shields.io/github/stars/ZhuLinsen/FastDatasets?style=social)](https://github.com/ZhuLinsen/FastDatasets/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mumu157/FastDatasets)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/fastdatasets-llm.svg)](https://pypi.org/project/fastdatasets-llm/)


一个强大的工具，用于为大语言模型（LLM）创建高质量的训练数据集 | [Switch to English](README_en.md)

## 🎯 在线体验

**🚀 立即体验 FastDatasets，无需安装！**

[![在 Hugging Face Spaces 上试用](https://img.shields.io/badge/🤗%20试用%20Demo-快速体验-orange?style=for-the-badge)](https://huggingface.co/spaces/mumu157/FastDatasets)

上传你的文档，一键生成 Alpaca 格式训练数据集 - 完全免费，无需配置环境！

| 使用方式 | 适用场景 | 特点 |
|---------|---------|------|
| 🤗 **Spaces 体验版** | 快速试用、功能演示 | 零配置、即开即用、仅供体验 |
| 💻 **本地完整版** | 生产环境、实际应用 | 无限制、批量处理、完整功能 |

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

### Web 界面使用

启动 Web 界面，提供可视化的文档处理和数据集生成功能：

```bash
# 进入 web 目录
cd web

# 启动 Web 应用
python web_app.py
```

默认地址：http://localhost:7860

#### Web 界面功能

1. **文件上传**：支持拖拽或点击上传多种格式的文档
   - 支持 PDF、Word、Markdown、纯文本等格式
   - 可同时上传多个文件进行批量处理

2. **参数配置**：
   - **文本分块设置**：配置最小/最大分块长度
   - **输出格式选择**：支持 Alpaca、ShareGPT 等格式
   - **LLM 配置**：设置 API Key、Base URL、模型名称
   - **并发控制**：调整 LLM 和文件处理的并发数
   - **高级选项**：启用思维链（CoT）、设置每块问题数量

3. **实时处理监控**：
   - 显示处理进度和当前状态
   - 实时更新处理日志
   - 显示已处理文件数量和剩余时间

4. **结果管理**：
   - 查看生成的问答对数量和质量
   - 下载生成的数据集文件
   - 支持多种格式的数据集导出

#### 使用步骤

1. **配置环境**：确保已正确配置 `.env` 文件中的 LLM API 信息
2. **启动服务**：运行 `python web_app.py` 启动 Web 界面
3. **上传文件**：在界面中上传需要处理的文档
4. **配置参数**：根据需要调整处理参数
5. **开始处理**：点击开始处理按钮，实时监控进度
6. **下载结果**：处理完成后下载生成的数据集

### 故障排除

1. **处理速度慢**：增加 MAX_LLM_CONCURRENCY 值，或减少 LLM_MAX_TOKENS
2. **内存不足**：减小处理的文档数量或文档大小
3. **API 超时**：检查网络连接，或减少 MAX_LLM_CONCURRENCY

## 许可证
[Apache 2.0](LICENSE)

## 通过 PyPI 安装与使用

### 安装

```bash
pip install fastdatasets-llm
# 可选功能：
# pip install 'fastdatasets-llm[web]'   # Web/UI/API
# pip install 'fastdatasets-llm[doc]'   # 更佳文档解析（textract）
# pip install 'fastdatasets-llm[all]'   # 全部可选能力
```

或安装最新开发版：

```bash
pip install git+https://github.com/ZhuLinsen/FastDatasets.git@main
```

### 配置 LLM（环境变量）

```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.example.com/v1"
export LLM_MODEL="your-model"
```

### 命令行使用（CLI）

```bash
# 从文件/目录生成数据集（安装 fastdatasets-llm 后，CLI 名称为 fastdatasets）
fastdatasets generate ./data/sample.txt -o ./output

# 多格式导出与 JSONL 输出
fastdatasets generate ./docs -o ./output -f alpaca,sharegpt --file-format jsonl
```

### Python 方式

```python
# 安装包名为 fastdatasets-llm，但导入名仍为 fastdatasets
from fastdatasets import generate_dataset_to_dir

dataset = generate_dataset_to_dir(
  inputs=["./data/sample.txt"],
  output_dir="./output",
  formats=["alpaca", "sharegpt"],
  file_format="jsonl",
  chunk_size=1000,
  chunk_overlap=200,
  enable_cot=False,
  max_llm_concurrency=5,
  # 如需覆盖 .env，可直接传：api_key="sk-...", api_base="https://api.example.com/v1", model_name="your-model"
)
print(f"Generated items: {len(dataset)}")
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZhuLinsen/FastDatasets&type=Date)](https://www.star-history.com/#ZhuLinsen/FastDatasets&Date)

