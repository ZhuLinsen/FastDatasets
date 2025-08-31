from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, List, Optional, Union

from app.core.config import Config, config
from app.core.document import DocumentProcessor
from app.core.dataset import DatasetBuilder


InputPaths = Union[str, Path, Iterable[Union[str, Path]]]


def _normalize_inputs(inputs: InputPaths) -> List[Path]:
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]  # type: ignore[list-item]
    paths: List[Path] = []
    for p in inputs:  # type: ignore[assignment]
        path = Path(p)
        if path.exists():
            paths.append(path)
    return paths


def _apply_overrides(
    api_key: Optional[str],
    api_base: Optional[str],
    model_name: Optional[str],
    enable_cot: Optional[bool],
    max_llm_concurrency: Optional[int],
    output_formats: Optional[List[str]],
) -> None:
    """Override the global runtime config in-place.

    Precedence: direct params (if provided) > environment variables (already loaded).
    """
    if api_key is not None:
        config.API_KEY = api_key
    if api_base is not None:
        config.BASE_URL = api_base
    if model_name is not None:
        config.MODEL_NAME = model_name
    if enable_cot is not None:
        config.ENABLE_COT = bool(enable_cot)
    if max_llm_concurrency is not None:
        config.MAX_LLM_CONCURRENCY = int(max_llm_concurrency)
    if output_formats is not None and output_formats:
        config.OUTPUT_FORMATS = output_formats


def generate_dataset(
    inputs: InputPaths,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 200,
    enable_cot: Optional[bool] = None,
    max_llm_concurrency: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[dict]:
    """Generate dataset from files/dirs with simple, sync API.

    - Hides asyncio usage internally
    - Allows overriding common knobs (chunk_size/overlap, COT, concurrency, LLM configs)
    """
    _apply_overrides(api_key, api_base, model_name, enable_cot, max_llm_concurrency, None)

    paths = _normalize_inputs(inputs)
    processor = DocumentProcessor()

    all_chunks: List[dict] = []
    for path in paths:
        if path.is_dir():
            for fp in path.rglob("*.*"):
                chunks = processor.process_document(str(fp), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                all_chunks.extend(chunks)
        else:
            chunks = processor.process_document(str(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)

    builder = DatasetBuilder()
    dataset: List[dict] = asyncio.run(builder.build_dataset(all_chunks))
    return dataset


def generate_dataset_to_dir(
    inputs: InputPaths,
    output_dir: Union[str, Path] = "output",
    *,
    formats: Optional[List[str]] = None,
    file_format: str = "json",
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 200,
    enable_cot: Optional[bool] = None,
    max_llm_concurrency: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[dict]:
    """Generate dataset and export to directory in chosen formats.

    Returns the in-memory dataset (list of dict) for further processing.
    """
    if formats is None:
        formats = ["alpaca"]

    # Apply overrides including output formats
    _apply_overrides(api_key, api_base, model_name, enable_cot, max_llm_concurrency, formats)

    dataset = generate_dataset(
        inputs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_cot=enable_cot,
        max_llm_concurrency=max_llm_concurrency,
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
    )

    builder = DatasetBuilder()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    builder.export_dataset(dataset, str(output_dir), formats=formats, file_format=file_format)
    return dataset


