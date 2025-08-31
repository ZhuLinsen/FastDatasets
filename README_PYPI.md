# FastDatasets

Generate high-quality LLM training datasets from documents. Distillation, augmentation, multi-format export.

## Install

```bash
pip install fastdatasets
# Optional extras:
# pip install 'fastdatasets[web]'   # Web UI / API
# pip install 'fastdatasets[doc]'   # Better doc parsing (textract)
# pip install 'fastdatasets[all]'   # Everything
```

## Configure LLM

Use environment variables or pass parameters directly (function args override env):

```bash
export LLM_API_KEY="sk-..."
export LLM_API_BASE="https://api.example.com/v1"
export LLM_MODEL="your-model"
```

## Quick Start (Python)

```python
from fastdatasets import generate_dataset_to_dir

dataset = generate_dataset_to_dir(
  inputs=["./docs", "./data/sample.txt"],
  output_dir="./output",
  formats=["alpaca", "sharegpt"],
  file_format="jsonl",
  chunk_size=1000,
  chunk_overlap=200,
  enable_cot=False,
  max_llm_concurrency=5,
  # api_key="sk-...", api_base="https://api.example.com/v1", model_name="your-model",
)
print(len(dataset))
```

## CLI

```bash
# Core usage
fastdatasets generate ./data -o ./output -f alpaca,sharegpt --file-format jsonl

# Override LLM just for this command
LLM_API_KEY=sk-xxx LLM_API_BASE=https://api.example.com/v1 LLM_MODEL=your-model \
  fastdatasets generate ./docs -o ./out
```

## Optional Features
- Web/API: `pip install 'fastdatasets[web]'` then run your web/app code
- Better doc parsing (PDF/DOCX): `pip install 'fastdatasets[doc]'`

## Links
- Source: https://github.com/ZhuLinsen/FastDatasets
- Demo (Spaces): https://huggingface.co/spaces/mumu157/FastDatasets
- Issues: https://github.com/ZhuLinsen/FastDatasets/issues
