# FastDatasets 🚀

![FastDatasets](fastdatasets.png)

[![GitHub Stars](https://img.shields.io/github/stars/ZhuLinsen/FastDatasets?style=social)](https://github.com/ZhuLinsen/FastDatasets/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/mumu157/FastDatasets)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/fastdatasets-llm.svg)](https://pypi.org/project/fastdatasets-llm/)


A powerful tool for creating high-quality training datasets for Large Language Models (LLMs) | [切换到中文](README.md)

## 🎯 Try Online

**🚀 Experience FastDatasets instantly, no installation required!**

[![Try on Hugging Face Spaces](https://img.shields.io/badge/🤗%20Try%20Demo-Experience%20Now-orange?style=for-the-badge)](https://huggingface.co/spaces/mumu157/FastDatasets)

Upload your documents and generate Alpaca-format training datasets with one click - completely free, no setup needed!

| Usage Mode | Use Case | Features |
|------------|----------|----------|
| 🤗 **Spaces Demo** | Quick trial, feature demo | Zero config, instant use, demo only |
| 💻 **Local Full Version** | Production, real usage | Unlimited, batch processing, complete features |

## Main Features

### 1. Document-Based Dataset Generation
- **Intelligent Document Processing**: Supports intelligent segmentation of various document formats
- **Question Generation**: Automatically generates relevant questions based on document content
- **Answer Generation**: Uses LLM to generate high-quality answers
- **Asynchronous Processing**: Supports asynchronous processing of large-scale documents
- **Multiple Export Formats**: Supports multiple dataset format exports (Alpaca, ShareGPT)
- **Direct SFT-Ready Output**: Generates datasets suitable for supervised fine-tuning

### 2. Data Distillation and Optimization
- **Dataset Distillation**: Extracts and optimizes training data from existing datasets
- **Instruction Augmentation**: Automatically generates instruction variants to expand training data
- **Quality Optimization**: Uses LLM to optimize and improve data quality
- **Multi-format Support**: Supports distillation from datasets in various formats

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
pip install -e .

# Create environment configuration file
cp .env.example .env
# Edit the .env file to configure your LLM API and other settings
```

### Usage Examples

1. First, configure your LLM
```bash
# Create and edit .env file by copying from .env.example
cp .env.example .env
# Edit .env file and configure your LLM API
# LLM_API_KEY=your_api_key_here
# LLM_API_BASE=https://api.deepseek.com/v1 (or other model API URL)

# Test LLM connection and capabilities
python scripts/test_llm.py
```

2. Generate dataset from documents
```bash
# Generate dataset using test document
python scripts/dataset_generator.py tests/test.txt -o ./output

# Process custom document
python scripts/dataset_generator.py path/to/your/document.pdf -o ./output_directory

# Process all documents in a directory
python scripts/dataset_generator.py path/to/document/directory/ -o ./output_directory
```

3. Create training datasets using knowledge distillation
```bash
# Extract questions from Huggingface dataset and generate high-quality answers
python scripts/distill_dataset.py --mode distill --dataset_name open-r1/s1K-1.1 --sample_size 10

# Generate variants from high-quality samples and distill knowledge (optional)
python scripts/distill_dataset.py --mode augment --high_quality_file data/high_quality_samples.json --num_aug 3
```

For more detailed usage, please refer to: [Knowledge Distillation Guide](docs/knowledge_distillation.md) | [Document Processing and Dataset Generation](docs/custom_data_conversion.md)

### Supported Document Formats

- PDF (*.pdf)
- Word (*.docx)
- Markdown (*.md)
- Plain text (*.txt)
- Other mainstream document formats

## Output Structure

### Document Processing Output
After processing, the following files are generated in the output directory:

- `chunks.json`: Text chunks after document segmentation
- `questions.json`: Generated questions
- `answers.json`: Generated answers
- `optimized.json`: Optimized QA pairs
- `dataset-alpaca.json`: Dataset in Alpaca format
- `dataset-sharegpt.json`: Dataset in ShareGPT format (if configured)

### Data Distillation Output
The data distillation process creates a timestamp subdirectory in the output directory and generates the following files:

- `distilled.json`: Original data after distillation
- `distilled-alpaca.json`: Distilled data in Alpaca format
- `distilled-sharegpt.json`: Distilled data in ShareGPT format

## Advanced Usage

### Web Interface Usage

Start the Web interface for visual document processing and dataset generation:

```bash
# Enter the web directory
cd web

# Start the Web application
python web_app.py
```

Default address: http://localhost:7860

#### Web Interface Features

1. **File Upload**: Support drag-and-drop or click to upload various document formats
   - Supports PDF, Word, Markdown, plain text, and other formats
   - Can upload multiple files for batch processing

2. **Parameter Configuration**:
   - **Text Chunking Settings**: Configure minimum/maximum chunk length
   - **Output Format Selection**: Support Alpaca, ShareGPT, and other formats
   - **LLM Configuration**: Set API Key, Base URL, model name
   - **Concurrency Control**: Adjust LLM and file processing concurrency
   - **Advanced Options**: Enable Chain of Thought (CoT), set questions per chunk

3. **Real-time Processing Monitoring**:
   - Display processing progress and current status
   - Real-time processing log updates
   - Show processed file count and remaining time

4. **Result Management**:
   - View generated QA pair count and quality
   - Download generated dataset files
   - Support multiple dataset format exports

#### Usage Steps

1. **Configure Environment**: Ensure proper configuration of LLM API information in `.env` file
2. **Start Service**: Run `python web_app.py` to start the Web interface
3. **Upload Files**: Upload documents to be processed in the interface
4. **Configure Parameters**: Adjust processing parameters as needed
5. **Start Processing**: Click the start processing button and monitor progress in real-time
6. **Download Results**: Download generated datasets after processing completion



### Troubleshooting

1. **Slow processing**: Increase MAX_LLM_CONCURRENCY or reduce LLM_MAX_TOKENS
2. **Out of memory**: Reduce the number or size of documents being processed
3. **API timeout**: Check network connection or reduce MAX_LLM_CONCURRENCY

## License
[Apache 2.0](LICENSE)

## Install and Use via PyPI

### Install

```bash
pip install fastdatasets-llm
# Optional extras:
# pip install 'fastdatasets-llm[web]'   # Web UI / API
# pip install 'fastdatasets-llm[doc]'   # Better doc parsing (textract)
# pip install 'fastdatasets-llm[all]'   # Everything
```

Or install the latest dev version:

```bash
pip install git+https://github.com/ZhuLinsen/FastDatasets.git@main
```

### Configure LLM (environment variables)

```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.example.com/v1"
export LLM_MODEL="your-model"
```

### Command Line (CLI)

```bash
# After installing fastdatasets-llm, the CLI name is still `fastdatasets`

# Generate dataset from file/directory
fastdatasets generate ./data/sample.txt -o ./output

# Multi-format export and JSONL output
fastdatasets generate ./docs -o ./output -f alpaca,sharegpt --file-format jsonl
```

### Python API

```python
# Distribution name is fastdatasets-llm, but the import name remains `fastdatasets`
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
  # To override .env, pass directly:
  # api_key="sk-...", api_base="https://api.example.com/v1", model_name="your-model"
)
print(f"Generated items: {len(dataset)}")
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZhuLinsen/FastDatasets&type=Date)](https://www.star-history.com/#ZhuLinsen/FastDatasets&Date)