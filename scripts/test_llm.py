import os
import sys
import json
import argparse
import asyncio
import httpx
import traceback
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from app.core.config import config

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# 加载环境变量
load_dotenv()

# 初始化控制台
console = Console()

async def test_llm_connection(api_key=None, base_url=None, model_name=None):
    """测试 LLM API 连接是否正常"""
    # 使用传入的参数或配置文件中的默认值
    api_key = api_key or os.getenv("API_KEY") or config.API_KEY
    base_url = base_url or os.getenv("BASE_URL") or config.BASE_URL
    model_name = model_name or os.getenv("MODEL_NAME") or config.MODEL_NAME
    
    console.print(Panel(
        f"API_KEY: {'*'*(len(api_key)-4) + api_key[-4:] if api_key else '未设置'}\n"
        f"BASE_URL: {base_url}\n"
        f"MODEL_NAME: {model_name}",
        title="LLM 连接配置",
        border_style="blue"
    ))
    
    # 确保 base_url 格式正确
    if base_url and not base_url.startswith(('http://', 'https://')):
        base_url = f"https://{base_url}"
    
    if not api_key or not base_url or not model_name:
        console.print(Panel("缺少必要的 LLM 配置参数", title="错误", border_style="red"))
        return False, "缺少必要的 LLM 配置参数"
    
    # 构建请求
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "请回复「测试成功」二字，不要有任何其他内容"}],
        "max_tokens": 10
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{base_url}/chat/completions", 
                                     headers=headers, 
                                     json=data, 
                                     follow_redirects=True)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            
            console.print(Panel(
                f"模型: {model_name}\nAPI Base: {base_url}\n响应: {content}",
                title="LLM 连接测试成功",
                border_style="green"
            ))
            return True, content
    except Exception as e:
        error_msg = str(e)
        console.print(Panel(
            f"错误信息: {error_msg}\n\n{traceback.format_exc()}",
            title="LLM 连接测试失败",
            border_style="red"
        ))
        return False, error_msg

async def test_llm_capabilities(api_key=None, base_url=None, model_name=None):
    """测试 LLM 的基本能力"""
    # 使用传入的参数或配置文件中的默认值
    api_key = api_key or os.getenv("API_KEY") or config.API_KEY
    base_url = base_url or os.getenv("BASE_URL") or config.BASE_URL
    model_name = model_name or os.getenv("MODEL_NAME") or config.MODEL_NAME
    
    # 确保 base_url 格式正确
    if base_url and not base_url.startswith(('http://', 'https://')):
        base_url = f"https://{base_url}"
    
    # 测试文本
    test_text = """
人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。它涉及开发能够感知、推理、学习和做决策的系统。人工智能的应用广泛，包括自然语言处理、计算机视觉、机器人技术和专家系统等领域。
"""
    
    # 构建请求
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": f"根据以下文本，生成一个高质量的问题。要求问题具有明确的指向性，能检验对文本核心内容的理解：\n{test_text}"}],
        "max_tokens": 500
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{base_url}/chat/completions", 
                                     headers=headers, 
                                     json=data, 
                                     follow_redirects=True)
            resp.raise_for_status()            
            print(f"=================\nresp.json(): {resp.json()}\n=================")
            content = resp.json()["choices"][0]["message"]["content"].strip()
            
            # 处理推理内容
            if config.ENABLE_REASONING_CONTENT and "reasoning_content" in resp.json()["choices"][0]["message"]:
                reasoning_content = resp.json()["choices"][0]["message"]["reasoning_content"].strip()
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            
            console.print("[bold green]LLM 功能测试成功！[/bold green]")
            console.print(Panel(content, title="生成的问题", border_style="green"))
            return True, content
    except Exception as e:
        console.print(f"[bold red]LLM 功能测试失败: {str(e)}[/bold red]")
        console.print(traceback.format_exc())
        return False, str(e)

async def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试 LLM API 连接")
    parser.add_argument("--api_key", help="API 密钥")
    parser.add_argument("--base_url", help="API 基础 URL")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--provider", help="LLM 提供商 (openai, azure, deepseek, zhipu, anthropic)")
    args = parser.parse_args()
    
    console.print("[bold blue]开始 LLM 配置测试[/bold blue]")
    
    # 如果指定了提供商，设置相应的默认值
    if args.provider:
        if args.provider.lower() == "openai":
            base_url = args.base_url or "https://api.openai.com/v1"
            model = args.model or "gpt-3.5-turbo"
        elif args.provider.lower() == "azure":
            base_url = args.base_url or os.getenv("AZURE_OPENAI_ENDPOINT", "")
            model = args.model or os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")
        elif args.provider.lower() == "deepseek":
            base_url = args.base_url or "https://api.deepseek.com/v1"
            model = args.model or "deepseek-chat"
        elif args.provider.lower() == "zhipu":
            base_url = args.base_url or "https://open.bigmodel.cn/api/paas/v4"
            model = args.model or "glm-4"
        elif args.provider.lower() == "anthropic":
            base_url = args.base_url or "https://api.anthropic.com/v1"
            model = args.model or "claude-3-opus-20240229"
        else:
            # 使用配置文件或环境变量
            base_url = args.base_url
            model = args.model
    else:
        # 使用配置文件或环境变量
        base_url = args.base_url
        model = args.model
    
    api_key = args.api_key
    
    # 测试 LLM 连接
    conn_success, conn_msg = await test_llm_connection(api_key, base_url, model)
    
    # 如果连接成功，测试功能
    if conn_success:
        cap_success, cap_msg = await test_llm_capabilities(api_key, base_url, model)
        
        if cap_success:
            console.print("[bold green]所有测试完成！LLM 配置工作正常[/bold green]")
            return 0
        else:
            console.print("[bold red]LLM 功能测试失败！[/bold red]")
            return 1
    else:
        console.print("[bold red]无法连接到 LLM API，请检查配置[/bold red]")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 