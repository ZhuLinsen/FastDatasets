# FastDatasets

- A powerful tool for creating high-quality training datasets for Large Language Models (LLMs)（一个快速生成高质量LLM训练数据集的工具）

[中文版](README_zh.md)

## Features

- **Intelligent Document Processing**: Smart segmentation of multiple document formats
- **Question Generation**: Automatic generation of relevant questions based on document content  
- **Answer Generation**: High-quality answers using LLM
- **Asynchronous Processing**: Supports large-scale document processing
- **Multiple Export Formats**: Supports various dataset formats (Alpaca, ShareGPT, etc.)
- **Direct SFT-ready Output**: Generates datasets ready for supervised fine-tuning
- **LLM Configuration Validation**: Built-in tool to test LLM API connections and capabilities
- **RESTful API**: Complete API interface

## Quick Start

### Requirements

- Python 3.8+
- Dependencies: See `requirements.txt`

### Installation

```bash
# Clone repository
git clone https://github.com/ZhuLinsen/FastDatasets.git
cd FastDatasets

# Create and activate virtual environment (optional)
conda create -n fast_datasets python=3.10
conda activate fast_datasets

# Install dependencies
pip install -r requirements.txt
```

### Usage Example

The project includes a test document. You can quickly test with:
1. edit .env file and configure your LLM
you can use the following command to test your LLM connection and capabilities:
```bash
python scripts/test_llm.py
```

2. run the following command to generate dataset from test document

```bash
# Generate dataset from test document
python scripts/main_async.py tests/1706.03762v7.pdf -o ./output
```

This command processes the "Attention is All You Need" paper (test document) and generates a dataset in the ./output directory.

### Custom Documents

```bash
# Process a single document
python scripts/main_async.py path/to/your/document.pdf -o ./output_directory

# Process all documents in a directory
python scripts/main_async.py path/to/document/directory/ -o ./output_directory
```

### Supported Formats

- PDF (*.pdf)
- Word (*.docx)
- Markdown (*.md)
- Plain text (*.txt)

## Output Structure

After processing, the following files are generated in the output directory:

- `chunks.json`: Text chunks after document segmentation
- `questions.json`: Generated questions
- `answers.json`: Generated answers
- `optimized.json`: Optimized QA pairs
- `dataset-alpaca.json`: Dataset in Alpaca format
- `dataset-sharegpt.json`: Dataset in ShareGPT format (if configured)

## Advanced Usage

### API Service

Start the API service, providing a Web interface and RESTful API:

```bash
python -m app.main
```

Default address: http://localhost:8000

### Troubleshooting

1. **Slow processing**: Increase MAX_LLM_CONCURRENCY or reduce LLM_MAX_TOKENS
2. **Out of memory**: Reduce the number or size of documents being processed
3. **API timeout**: Check network connection or reduce MAX_LLM_CONCURRENCY

## License
[Apache 2.0](LICENSE)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZhuLinsen/FastDatasets&type=Date)](https://www.star-history.com/#ZhuLinsen/FastDatasets&Date)

