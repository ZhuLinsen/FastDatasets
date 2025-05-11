# 预留LLM相关接口，便于后续对接不同大模型

import asyncio
import httpx
import random
import time
import logging
import traceback
import os
from app.core.config import config
from app.core.logger import logger

class AsyncLLM:
    def __init__(self, model_name=None, base_url=None, api_key=None, language=None, max_concurrency=None, system_prompt=None):
        self.model_name = model_name or config.MODEL_NAME
        self.base_url = base_url or config.BASE_URL
        self.api_key = api_key or config.API_KEY
        self.language = language or config.LANGUAGE
        self.max_concurrency = max_concurrency or config.MAX_LLM_CONCURRENCY
        self.system_prompt = system_prompt or getattr(config, 'SYSTEM_PROMPT', None)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    # 保持原有的简单接口，但内部使用高级实现
    async def call_llm(self, prompt, max_tokens=2048*2):
        """原始的简单LLM调用接口，保持向后兼容"""
        return await self.call_llm_advanced(prompt=prompt, max_tokens=max_tokens)
    
    async def call_llm_advanced(self, prompt, max_tokens=2048*2, retries=8, backoff_factor=1.8, 
                                 dynamic_timeout=True, return_exceptions=False):
        """高级LLM调用接口，支持错误处理、重试机制、动态超时等功能"""
        # 异步信号量控制
        async with self.semaphore:
            # 重新从环境变量获取配置，确保使用最新设置
            api_key = os.getenv("API_KEY") or self.api_key
            base_url = os.getenv("BASE_URL") or self.base_url
            model_name = os.getenv("MODEL_NAME") or self.model_name
            
            # 更新当前实例的设置
            self.api_key = api_key
            self.base_url = base_url
            self.model_name = model_name
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # 确保 API URL 格式正确
            if self.base_url and not self.base_url.startswith(('http://', 'https://')):
                self.base_url = f"https://{self.base_url}"
                
            # 检查必要参数
            if not self.api_key or not self.base_url or not self.model_name:
                logger.error("缺少必要的 LLM 配置参数")
                if return_exceptions:
                    return RuntimeError("缺少必要的LLM配置参数")
                return self._fallback_response(prompt)
            
            # 动态超时设置 - 根据prompt长度调整
            if dynamic_timeout:
                base_timeout = 60
                timeout_per_token = 0.06  # 每token增加的超时时间(秒)
                estimated_tokens = len(prompt) / 3  # 估算token数量
                timeout = base_timeout + min(480, estimated_tokens * timeout_per_token)  # 最多增加480秒(8分钟)
            else:
                timeout = 120*10  # 默认超时
            
            # 生成一个请求ID用于日志追踪
            request_id = f"req-{random.randint(1000, 9999)}"
            
            for attempt in range(retries):
                try:
                    # 随机化超时时间，避免所有请求同时超时
                    jitter = 1.0 + random.uniform(-0.15, 0.15)  # 随机因子±15%
                    current_timeout = timeout * jitter
                    if attempt > 0:
                        current_timeout *= (1 + attempt * 0.6)  # 每次重试增加60%超时时间
                    
                    logger.debug(f"[{request_id}] API调用超时设置: {current_timeout:.1f}秒 (尝试 {attempt+1}/{retries})")
                    
                    # 准备请求数据
                    data = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens
                    }
                    
                    # 可选: 添加系统提示
                    if self.system_prompt:
                        data["messages"].insert(0, {"role": "system", "content": self.system_prompt})
                    
                    # 可选: 调整温度等参数
                    if hasattr(config, 'TEMPERATURE') and config.TEMPERATURE is not None:
                        data["temperature"] = float(config.TEMPERATURE)
                    
                    if hasattr(config, 'TOP_P') and config.TOP_P is not None:
                        data["top_p"] = float(config.TOP_P)
                    
                    # 日志记录开始信息 - 避免记录完整提示内容，只记录前30个字符
                    prompt_preview = prompt[:30].replace('\n', ' ') + "..." if len(prompt) > 30 else prompt
                    logger.debug(f"[{request_id}] 发送请求到 {self.model_name} (尝试 {attempt+1}/{retries})")
                    logger.debug(f"[{request_id}] 提示预览: {prompt_preview}")
                    
                    start_time = time.time()
                    
                    # 使用异步上下文管理器创建客户端
                    async with httpx.AsyncClient(timeout=current_timeout) as client:
                        try:
                            # 发送请求
                            resp = await client.post(
                                f"{self.base_url}/chat/completions", 
                                headers=self.headers, 
                                json=data, 
                                follow_redirects=True
                            )
                            
                            # 检查状态码
                            resp.raise_for_status()
                            elapsed = time.time() - start_time
                            
                            # 解析响应
                            response_json = resp.json()
                            content = response_json["choices"][0]["message"]["content"]
                            
                            # 查找可能的token使用情况信息
                            token_info = ""
                            if "usage" in response_json:
                                usage = response_json["usage"]
                                token_info = f", 使用了 {usage.get('prompt_tokens', 0)} 输入token, {usage.get('completion_tokens', 0)} 输出token"
                            
                            logger.debug(f"[{request_id}] 请求成功，耗时 {elapsed:.2f}秒，输出长度: {len(content)}{token_info}")
                            
                            # 可选: 显示输出前几个字符用于调试
                            if isinstance(logging.getLogger().level, int) and logging.getLogger().level <= logging.DEBUG:
                                preview = content[:50].replace('\n', ' ') + "..." if len(content) > 50 else content
                                logger.debug(f"[{request_id}] 输出预览: {preview}")
                                
                            return content.strip()
                            
                        except httpx.HTTPStatusError as e:
                            elapsed = time.time() - start_time
                            status_code = e.response.status_code
                            error_text = e.response.text[:200] + "..." if len(e.response.text) > 200 else e.response.text
                            
                            logger.error(f"[{request_id}] HTTP 错误 ({elapsed:.2f}秒): {status_code} - {error_text}")
                            
                            if status_code == 401:
                                logger.error(f"[{request_id}] API 密钥错误或未授权")
                                if return_exceptions:
                                    return httpx.HTTPStatusError(f"认证错误: {error_text}", request=e.request, response=e.response)
                                break  # 认证错误不重试
                                
                            elif status_code == 429:
                                logger.warning(f"[{request_id}] 请求频率限制，将重试")
                                # 对于频率限制错误，使用更长的等待时间
                                wait_time = backoff_factor * (2.5 ** attempt)
                                logger.warning(f"[{request_id}] 等待 {wait_time:.1f} 秒后重试...")
                                await asyncio.sleep(wait_time)
                                continue
                                
                            elif status_code >= 500:
                                logger.warning(f"[{request_id}] 服务器错误 ({status_code})，将重试")
                                wait_time = backoff_factor * (2 ** attempt)
                                logger.warning(f"[{request_id}] 等待 {wait_time:.1f} 秒后重试...")
                                await asyncio.sleep(wait_time)
                                continue
                                
                            # 其他HTTP错误
                            if attempt < retries - 1:
                                wait_time = backoff_factor * (2 ** attempt)
                                logger.warning(f"[{request_id}] 等待 {wait_time:.1f} 秒后重试...")
                                await asyncio.sleep(wait_time)
                            else:
                                logger.error(f"[{request_id}] 已达到最大重试次数")
                                if return_exceptions:
                                    return e
                                break
                        
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    logger.warning(f"[{request_id}] 连接/读取错误 ({type(e).__name__}): {str(e)} ({elapsed:.1f}秒)")
                    
                    if attempt < retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"[{request_id}] 将在 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"[{request_id}] 连接失败，已达到最大重试次数: {str(e)}")
                        if return_exceptions:
                            return e
                        
                except Exception as e:
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    logger.error(f"[{request_id}] 调用 LLM API 失败 ({elapsed:.1f}秒): {str(e)}")
                    
                    if isinstance(logging.getLogger().level, int) and logging.getLogger().level <= logging.DEBUG:
                        logger.debug(f"[{request_id}] 异常详情: {traceback.format_exc()}")
                    
                    if attempt < retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"[{request_id}] 将在 {wait_time:.1f} 秒后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"[{request_id}] 已达到最大重试次数")
                        if return_exceptions:
                            return e
            
            # 所有重试都失败，返回后备响应
            logger.error(f"[{request_id}] 所有 API 调用尝试都失败，返回后备响应")
            if return_exceptions:
                return RuntimeError(f"所有API调用尝试都失败({retries}次)")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """当 LLM API 调用失败时的后备响应"""
        logger.warning("使用模拟回复代替 LLM 响应")
        if "question" in prompt.lower():
            return '["这是一个示例问题？", "这是另一个示例问题？"]'
        elif "answer" in prompt.lower():
            return "这是一个示例回答，由于无法连接到 LLM API 而生成的模拟内容。"
        else:
            return "模拟 LLM 响应"

# For sync usage or testing
class DummyLLM:
    def generate(self, prompt: str) -> str:
        return f"LLM output: {prompt}"

# 便于后续切换不同LLM
llm = AsyncLLM() 