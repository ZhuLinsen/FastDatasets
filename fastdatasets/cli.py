import argparse
import asyncio
import os
from pathlib import Path
from typing import List

from app.core.config import Config
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetBuilder


def run_generate(input_paths: List[str], output_dir: str, formats: List[str], file_format: str):
    cfg = Config()

    # 环境变量覆盖（便于 CLI 直接注入）
    api_key = os.getenv("LLM_API_KEY", cfg.API_KEY)
    base_url = os.getenv("LLM_API_BASE", cfg.BASE_URL)
    model = os.getenv("LLM_MODEL", cfg.MODEL_NAME)
    cfg.API_KEY, cfg.BASE_URL, cfg.MODEL_NAME = api_key, base_url, model

    processor = DocumentProcessor()
    builder = DatasetBuilder()

    # 收集所有块
    all_chunks = []
    for p in input_paths:
        path = Path(p)
        if path.is_dir():
            for fp in path.rglob("*.*"):
                chunks = processor.process_document(str(fp))
                all_chunks.extend(chunks)
        else:
            chunks = processor.process_document(str(path))
            all_chunks.extend(chunks)

    dataset = asyncio.run(builder.build_dataset(all_chunks))

    # 导出
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    builder.export_dataset(dataset, output_dir, formats=formats, file_format=file_format)


def main():
    parser = argparse.ArgumentParser(prog="fastdatasets", description="Generate LLM training datasets from documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate dataset from files or directories")
    gen.add_argument("inputs", nargs="+", help="Input file(s) or directory(ies)")
    gen.add_argument("-o", "--output", default="output", help="Output directory")
    gen.add_argument("-f", "--formats", default="alpaca", help="Export formats, comma-separated (alpaca,sharegpt)")
    gen.add_argument("--file-format", default="json", choices=["json", "jsonl"], help="Output file format")

    args = parser.parse_args()

    if args.command == "generate":
        formats = [s.strip() for s in str(args.formats).split(",") if s.strip()]
        run_generate(args.inputs, args.output, formats=formats, file_format=args.file_format)


if __name__ == "__main__":
    main()




