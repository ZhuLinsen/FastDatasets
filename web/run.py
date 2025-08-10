#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastDatasets Web Interface 启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import gradio
        print("✅ Gradio 已安装")
    except ImportError:
        print("❌ Gradio 未安装，正在安装依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成")

def main():
    """主函数"""
    print("🚀 启动 FastDatasets Web 界面...")
    
    # 检查依赖
    check_dependencies()
    
    # 设置环境变量
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    
    # 启动应用
    try:
        from web_app import create_interface
        
        interface = create_interface()
        
        print("\n" + "="*50)
        print("🎉 FastDatasets Web 界面已启动！")
        print("📱 访问地址: http://localhost:7860")
        print("🛑 按 Ctrl+C 停止服务")
        print("="*50 + "\n")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()