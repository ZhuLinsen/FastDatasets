import os
import sys
import json
from pathlib import Path
from typing import Optional
import gradio as gr


# Add parent directory to path if running locally
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def _fallback_demo(file_path: Optional[str], error_detail: str) -> str:
    """Fallback implementation when FastDatasets can't be imported"""
    import openai
    import os
    
    # Check if we have API key
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        return json.dumps({
            "error": "Cannot import FastDatasets and no LLM_API_KEY provided",
            "detail": error_detail,
            "hint": "Set LLM_API_KEY in Space secrets for fallback mode"
        }, ensure_ascii=False, indent=2)
    
    # Simple fallback implementation
    src = file_path or str(Path("samples/mini.txt").resolve())
    if not Path(src).exists():
        return json.dumps({"error": "input file not found", "path": src}, ensure_ascii=False, indent=2)
    
    try:
        # Read content (limit to 2000 chars)
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read(2000)
        
        # Simple OpenAI client setup
        base_url = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").strip()
        # Remove any trailing slashes and ensure clean URL
        base_url = base_url.rstrip('/')
        if not base_url.endswith('/v1'):
            if base_url.endswith('/'):
                base_url = base_url + 'v1'
            else:
                base_url = base_url + '/v1'
        
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # Generate questions
        question_prompt = f"""Based on this text, generate 2 questions that could be answered using the information:

Text: {content}

Return only a JSON array of questions, like: ["question1", "question2"]"""
        
        model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo").strip()
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": question_prompt}],
            max_tokens=100
        )
        
        questions_text = response.choices[0].message.content.strip()
        
        # Try to parse questions
        try:
            questions = json.loads(questions_text)
        except:
            # Fallback if JSON parsing fails
            questions = ["What is the main topic of this text?", "What are the key points mentioned?"]
        
        # Generate answers for each question
        alpaca_data = []
        for i, question in enumerate(questions[:2]):  # Limit to 2
            answer_prompt = f"""Based on this text, answer the question:

Text: {content}

Question: {question}

Provide a clear, concise answer based only on the information in the text."""
            
            answer_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": answer_prompt}],
                max_tokens=150
            )
            
            answer = answer_response.choices[0].message.content.strip()
            
            alpaca_data.append({
                "instruction": question,
                "input": "",
                "output": answer
            })
        
        return json.dumps(alpaca_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        import traceback
        return json.dumps({
            "error": "Fallback demo failed",
            "detail": str(e),
            "traceback": traceback.format_exc(),
            "debug_info": {
                "api_key_length": len(api_key) if api_key else 0,
                "base_url": base_url if 'base_url' in locals() else "not_set",
                "model_name": model_name if 'model_name' in locals() else "not_set"
            },
            "note": "This is a simplified version. Install FastDatasets for full features."
        }, ensure_ascii=False, indent=2)


def _run_real_call(file_path: Optional[str]) -> str:
    """Run a real call with strict limits"""
    import os
    import tempfile
    import asyncio
    import traceback
    
    Config = None
    DocumentProcessor = None
    DatasetBuilder = None
    import_error = None
    
    try:
        # Try different import strategies
        try:
            from app.core.config import Config
            from app.core.document import DocumentProcessor
            from app.core.dataset import DatasetBuilder
        except ImportError:
            # Try importing from installed package
            import fastdatasets
            from fastdatasets.app.core.config import Config
            from fastdatasets.app.core.document import DocumentProcessor
            from fastdatasets.app.core.dataset import DatasetBuilder
    except Exception as e:
        import_error = str(e)
        tb = traceback.format_exc()
    
    if import_error or not all([Config, DocumentProcessor, DatasetBuilder]):
        # Fallback to a simple mock implementation for demo purposes
        return _fallback_demo(file_path, import_error)

    # Pick input file: uploaded or tiny sample
    src = file_path or str(Path("samples/mini.txt").resolve())
    if not Path(src).exists():
        return json.dumps({"error": "input file not found", "path": src}, ensure_ascii=False, indent=2)

    # Configure with limits
    cfg = Config()
    
    # Apply environment overrides (strip whitespace/newlines)
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_API_BASE", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    
    if api_key:
        cfg.API_KEY = api_key
    if base_url:
        cfg.BASE_URL = base_url
    if model:
        cfg.MODEL_NAME = model
    
    # Ensure API key is present for real LLM calls
    if not api_key and not getattr(cfg, "API_KEY", None):
        return json.dumps({
            "error": "LLM_API_KEY missing in Space Secrets",
            "hint": "Set LLM_API_KEY (and optionally LLM_API_BASE, LLM_MODEL) in Settings → Repository secrets",
        }, ensure_ascii=False, indent=2)
    
    # Apply hard limits
    cfg.MAX_LLM_CONCURRENCY = 1
    if hasattr(cfg, "CHUNK_MIN_LEN"):
        cfg.CHUNK_MIN_LEN = 300
    if hasattr(cfg, "CHUNK_MAX_LEN"):
        cfg.CHUNK_MAX_LEN = 600
    if hasattr(cfg, "ENABLE_COT"):
        cfg.ENABLE_COT = False

    try:
        # Read and limit file content to 2000 chars
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read(2000)  # Hard limit at file read
        
        # Create a temporary file with limited content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            doc = DocumentProcessor()
            
            # Process only the limited content
            chunks = doc.process_document(tmp_path)
            
            # Take only first 2 chunks
            limited_chunks = chunks[:2] if chunks else []
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        
        builder = DatasetBuilder()
        
        # Use synchronous processing
        qa_pairs = asyncio.run(builder.build_dataset(limited_chunks))
        
        # Convert to Alpaca format
        alpaca_data = []
        for qa in qa_pairs[:5]:  # Limit to 5 for demo
            if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                alpaca_item = {
                    "instruction": qa['question'],
                    "input": "",
                    "output": qa['answer']
                }
                alpaca_data.append(alpaca_item)
        
        return json.dumps(alpaca_data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        # Return explicit error to help debugging on Space
        return json.dumps({
            "error": "processing failed",
            "detail": str(e),
            "trace": traceback.format_exc()
        }, ensure_ascii=False, indent=2)


with gr.Blocks() as app:
    gr.Markdown("# 🚀 FastDatasets Demo")
    gr.Markdown(
        "Transform your documents into high-quality LLM training datasets!\n\n"
        "📌 **Demo Limits**: Only first **2000 characters** processed, max 2-5 QA pairs output\n"
        "💡 **Smart Fallback**: Uses simplified processing if full package unavailable\n"
        "🔗 **Full version**: [GitHub](https://github.com/ZhuLinsen/FastDatasets) - No limits!"
    )

    with gr.Row():
        with gr.Column():
            up = gr.File(label="Upload a small text/markdown file (≤2MB)", file_types=[".txt", ".md"], file_count="single")
            run_real = gr.Button("🚀 Generate Dataset", variant="primary")
            run_pre = gr.Button("📄 Show Example", variant="secondary")
        with gr.Column():
            out = gr.Code(label="Generated Dataset (Alpaca Format)", language="json")

    run_pre.click(fn=_load_precomputed, inputs=None, outputs=out)
    run_real.click(fn=lambda f: _run_real_call(f.name if f else None), inputs=up, outputs=out)


if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", server_port=7860)


