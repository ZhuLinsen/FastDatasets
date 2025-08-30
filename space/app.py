import os
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
import gradio as gr

try:
    # Import project modules installed via pip (git+https://github.com/ZhuLinsen/FastDatasets.git)
    from app.core.config import Config
    from app.core.document import DocumentProcessor
    from app.core.dataset import DatasetBuilder
except Exception as e:
    Config = None
    DocumentProcessor = None
    DatasetBuilder = None


DEMO_TEXT = (
    "This is a tiny demo for FastDatasets on Hugging Face Spaces.\n"
    "We will either show a precomputed sample or run a tiny real call with strict limits."
)


def _load_precomputed() -> str:
    pre = Path("samples/precomputed/dataset-alpaca.json")
    if pre.exists():
        try:
            return pre.read_text(encoding="utf-8")
        except Exception:
            pass
    return DEMO_TEXT


def _apply_limited_config(cfg: Config):
    # Read secrets from Space and apply strict limits
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_API_BASE", "")
    model = os.getenv("LLM_MODEL", "")

    if api_key:
        cfg.API_KEY = api_key
    if base_url:
        cfg.BASE_URL = base_url
    if model:
        cfg.MODEL_NAME = model

    # Hard limits for demo
    cfg.MAX_LLM_CONCURRENCY = int(os.getenv("MAX_LLM_CONCURRENCY", "1"))
    # If your config supports token limits, set a small value via env
    if hasattr(cfg, "LLM_MAX_TOKENS"):
        try:
            cfg.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
        except Exception:
            pass

    # Keep chunks small so the demo is fast and cheap
    if hasattr(cfg, "CHUNK_MIN_LEN"):
        cfg.CHUNK_MIN_LEN = 300
    if hasattr(cfg, "CHUNK_MAX_LEN"):
        cfg.CHUNK_MAX_LEN = 600
    if hasattr(cfg, "ENABLE_COT"):
        cfg.ENABLE_COT = False


async def _run_real_call(file_path: Optional[str]) -> str:
    if not all([Config, DocumentProcessor, DatasetBuilder]):
        return _load_precomputed()

    cfg = Config()
    _apply_limited_config(cfg)

    # Pick input file: uploaded or tiny sample
    src = file_path or str(Path("samples/mini.txt").resolve())
    if not Path(src).exists():
        return _load_precomputed()

    try:
        doc = DocumentProcessor()
        chunks = doc.process_document(src) or []

        # Limit total readable length to 2000 characters across chunks
        def limit_chunks_by_chars(raw_chunks: List[Any], max_chars: int = 2000) -> List[Any]:
            total = 0
            limited: List[Any] = []
            for ch in raw_chunks:
                # detect text
                text = None
                text_key = None
                if isinstance(ch, dict):
                    for k in ("text", "content", "value", "chunk"):
                        if k in ch and isinstance(ch[k], str):
                            text = ch[k]
                            text_key = k
                            break
                elif isinstance(ch, str):
                    text = ch
                else:
                    # unsupported type; skip
                    continue

                remaining = max_chars - total
                if remaining <= 0:
                    break

                use_text = text[:remaining]
                total += len(use_text)

                if isinstance(ch, dict):
                    new_ch = dict(ch)
                    if text_key:
                        new_ch[text_key] = use_text
                    limited.append(new_ch)
                else:
                    limited.append(use_text)

                if total >= max_chars:
                    break
            return limited

        limited_by_chars = limit_chunks_by_chars(chunks, max_chars=2000)

        # Also cap number of chunks if specified
        max_chunks = int(os.getenv("MAX_CHUNKS", "2"))
        limited_chunks = limited_by_chars[:max_chunks]

        builder = DatasetBuilder()
        qa_pairs = await builder.build_dataset(limited_chunks)

        # Return a preview JSON (limit output size)
        preview = qa_pairs[: min(len(qa_pairs), 10)]
        return json.dumps(preview, ensure_ascii=False, indent=2)
    except Exception as e:
        # Fallback to precomputed on any error
        return _load_precomputed()


with gr.Blocks() as app:
    gr.Markdown("# FastDatasets – Tiny Real Demo (Cost-limited)")
    gr.Markdown(
        "Upload a small text/PDF/MD/DOCX (≤2MB) or use the built-in tiny sample.\n"
        "For cost control, we strictly limit chunks/tokens/concurrency."
    )

    with gr.Row():
        with gr.Column():
            up = gr.File(label="Upload a small file (≤2MB)", file_types=[".txt", ".pdf", ".md", ".docx"], file_count="single")
            run_real = gr.Button("Run tiny real call")
            run_pre = gr.Button("Show precomputed sample")
        with gr.Column():
            out = gr.Code(label="Output (Alpaca-like JSON)")

    run_pre.click(fn=_load_precomputed, inputs=None, outputs=out)
    run_real.click(fn=lambda f: asyncio.run(_run_real_call(f.name if f else None)), inputs=up, outputs=out)


if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", server_port=7860)


