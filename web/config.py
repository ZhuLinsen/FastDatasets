#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web界面配置文件
"""

import os
from pathlib import Path
from typing import List

class WebConfig:
    """Web界面配置类"""
    
    def __init__(self):
        # 项目路径
        self.PROJECT_ROOT = Path(__file__).parent.parent.absolute()
        
        # Web服务配置
        self.HOST = os.getenv("WEB_HOST", "0.0.0.0")
        self.PORT = int(os.getenv("WEB_PORT", "7860"))
        self.SHARE = os.getenv("WEB_SHARE", "false").lower() == "true"
        self.INBROWSER = os.getenv("WEB_INBROWSER", "true").lower() == "true"
        
        # 文件上传配置
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100"))  # MB
        self.SUPPORTED_FORMATS = [".pdf", ".docx", ".txt", ".md"]
        
        # 处理配置
        self.DEFAULT_OUTPUT_DIR = str(self.PROJECT_ROOT / "output")
        self.DEFAULT_CHUNK_MIN_LEN = 200
        self.DEFAULT_CHUNK_MAX_LEN = 1000
        self.DEFAULT_OUTPUT_FORMATS = ["alpaca", "sharegpt"]
        self.DEFAULT_ENABLE_COT = True
        self.DEFAULT_LLM_CONCURRENCY = 3
        self.DEFAULT_FILE_CONCURRENCY = 2
        
        # 界面配置
        self.THEME = "soft"  # soft, default, monochrome
        self.AUTO_REFRESH_INTERVAL = 3  # 秒
        self.RESULTS_REFRESH_INTERVAL = 5  # 秒
        
        # 任务管理
        self.MAX_CONCURRENT_TASKS = 5
        self.TASK_TIMEOUT = 3600  # 秒
        
    def get_gradio_theme(self):
        """获取Gradio主题"""
        import gradio as gr
        
        theme_map = {
            "soft": gr.themes.Soft(),
            "default": gr.themes.Default(),
            "monochrome": gr.themes.Monochrome()
        }
        
        return theme_map.get(self.THEME, gr.themes.Soft())
    
    def get_custom_css(self) -> str:
        """获取自定义CSS样式"""
        return """
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
        }
        
        .status-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        
        .upload-area {
            border: 2px dashed #007bff;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            background-color: #f8f9ff;
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #007bff 0%, #28a745 100%);
            height: 8px;
            border-radius: 4px;
        }
        
        .task-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .success-text {
            color: #28a745;
            font-weight: bold;
        }
        
        .error-text {
            color: #dc3545;
            font-weight: bold;
        }
        
        .warning-text {
            color: #ffc107;
            font-weight: bold;
        }
        
        .info-text {
            color: #17a2b8;
            font-weight: bold;
        }
        
        .tab-nav {
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .task-list {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .results-detail {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 10px 0;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        
        .progress-overview {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .progress-detail {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .progress-section {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metric-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .auto-refresh-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #28a745;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .dataset-preview {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
        }
        """

# 创建全局配置实例
web_config = WebConfig()