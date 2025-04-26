# 预留LLM相关接口，便于后续对接不同大模型

import asyncio
import httpx
from app.core.config import config
from app.core.logger import logger

class AsyncLLM:
    def __init__(self, model_name=None, base_url=None, api_key=None, language=None, max_concurrency=None):
        self.model_name = model_name or config.MODEL_NAME
        self.base_url = base_url or config.BASE_URL
        self.api_key = api_key or config.API_KEY
        self.language = language or config.LANGUAGE
        self.semaphore = asyncio.Semaphore(max_concurrency or config.MAX_LLM_CONCURRENCY)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def call_llm(self, prompt, max_tokens=2048*2):
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=120*10) as client:
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                }
                resp = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=data)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return content.strip()

# For sync usage or testing
class DummyLLM:
    def generate(self, prompt: str) -> str:
        return f"LLM output: {prompt}"

# 便于后续切换不同LLM
llm = AsyncLLM() 