# FastDatasets

一个强大的工具，用于为大语言模型（LLM）创建高质量的训练数据集 ([Switch to English](README.md))
![FastDatasets](fastdatasets.png)

## 主要特性

- **智能文档处理**：支持多种格式文档的智能分割
- **问题生成**：基于文档内容自动生成相关问题
- **答案生成**：使用 LLM 生成高质量答案
- **异步处理**：支持大规模文档的异步处理
- **多种导出格式**：支持多种数据集格式导出（Alpaca、ShareGPT等）
- **直接SFT就绪输出**：生成适用于监督微调的数据集
- **LLM配置验证**：内置工具测试LLM API连接和能力
- **RESTful API**：提供完整的 API 接口

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
pip install -r requirements.txt
```

### 使用示例

项目已包含一个测试文档，您可以快速测试：
1. 编辑.env文件并配置您的LLM
   您可以使用以下命令测试LLM连接和能力：
```bash
python scripts/test_llm.py
```

2. 运行以下命令从测试文档生成数据集

```bash
# 使用测试文档生成数据集
python scripts/main_async.py tests/1706.03762v7.pdf -o ./output
```

这个命令会处理"Attention is All You Need"论文（测试文档），并在 ./output 目录下生成数据集。

### 处理自定义文档

```bash
# 处理单个文档
python scripts/main_async.py 路径/到/你的文档.pdf -o ./输出目录

# 处理整个目录下的文档
python scripts/main_async.py 路径/到/文档目录/ -o ./输出目录
```

### 支持的文档格式

- PDF (*.pdf)
- Word (*.docx)
- Markdown (*.md)
- 纯文本 (*.txt)

## 输出结构

处理完成后，在指定的输出目录会生成以下文件：

- `chunks.json`: 文档切分后的文本块
- `questions.json`: 生成的问题
- `answers.json`: 生成的答案
- `optimized.json`: 优化后的问答对
- `dataset-alpaca.json`: Alpaca 格式的数据集
- `dataset-sharegpt.json`: ShareGPT 格式的数据集（如果配置了此输出格式）

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