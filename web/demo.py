#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastDatasets Web Interface 演示脚本
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

def create_demo_files():
    """创建演示文件"""
    demo_dir = project_root / "demo_files"
    demo_dir.mkdir(exist_ok=True)
    
    # 创建示例Markdown文件
    md_content = """
# 人工智能基础知识

## 什么是人工智能？

人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。

### 主要特征

1. **学习能力**：AI系统能够从数据中学习并改进性能
2. **推理能力**：能够基于已知信息得出结论
3. **感知能力**：能够理解和解释环境信息
4. **决策能力**：能够在复杂情况下做出最优选择

## 机器学习类型

### 监督学习

监督学习使用标记的训练数据来学习输入和输出之间的映射关系。

**应用场景**：
- 图像分类
- 语音识别
- 文本分类
- 预测分析

### 无监督学习

无监督学习从未标记的数据中发现隐藏的模式和结构。

**常见算法**：
- 聚类算法（K-means、层次聚类）
- 降维算法（PCA、t-SNE）
- 关联规则学习

### 强化学习

强化学习通过与环境交互，通过奖励和惩罚机制来学习最优策略。

**核心概念**：
- 智能体（Agent）
- 环境（Environment）
- 状态（State）
- 动作（Action）
- 奖励（Reward）

## 深度学习

深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的学习过程。

### 神经网络架构

1. **前馈神经网络**：信息单向流动的基础网络
2. **卷积神经网络（CNN）**：专门用于处理图像数据
3. **循环神经网络（RNN）**：适合处理序列数据
4. **Transformer**：基于注意力机制的现代架构

### 应用领域

- **计算机视觉**：图像识别、目标检测、图像生成
- **自然语言处理**：机器翻译、文本生成、情感分析
- **语音技术**：语音识别、语音合成
- **推荐系统**：个性化推荐、内容过滤

## AI的发展历程

### 第一次AI浪潮（1950s-1960s）
- 符号主义AI的兴起
- 专家系统的发展
- 图灵测试的提出

### 第二次AI浪潮（1980s-1990s）
- 机器学习算法的发展
- 神经网络的复兴
- 统计学习理论的建立

### 第三次AI浪潮（2010s-至今）
- 深度学习的突破
- 大数据和计算能力的提升
- 预训练模型的普及

## 未来展望

人工智能技术将继续快速发展，预计在以下领域取得重大突破：

1. **通用人工智能（AGI）**：具备人类级别智能的AI系统
2. **多模态AI**：能够处理文本、图像、音频等多种数据类型
3. **可解释AI**：提供决策过程透明度的AI系统
4. **边缘AI**：在设备端运行的轻量级AI模型
5. **AI伦理**：确保AI技术的安全、公平和负责任使用

## 总结

人工智能作为21世纪最重要的技术之一，正在深刻改变我们的生活和工作方式。理解AI的基本概念、技术原理和应用场景，对于在AI时代保持竞争力至关重要。
"""
    
    with open(demo_dir / "ai_basics.md", 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 创建示例文本文件
    txt_content = """
深度学习框架比较

TensorFlow
TensorFlow是Google开发的开源机器学习框架，具有以下特点：
- 强大的生态系统和社区支持
- 支持多种编程语言（Python、C++、Java等）
- 优秀的生产部署能力
- TensorBoard可视化工具
- 支持分布式训练

PyTorch
PyTorch是Facebook开发的深度学习框架，主要优势包括：
- 动态计算图，调试友好
- Pythonic的API设计
- 强大的研究社区
- 灵活的模型构建方式
- 优秀的GPU加速支持

Keras
Keras是高级神经网络API，现已集成到TensorFlow中：
- 简洁易用的API
- 快速原型开发
- 支持多种后端
- 丰富的预训练模型
- 良好的文档和教程

选择建议
对于初学者，推荐从Keras开始学习，因为其API简洁易懂。
对于研究人员，PyTorch提供了更大的灵活性。
对于生产环境，TensorFlow具有更成熟的部署工具。
"""
    
    with open(demo_dir / "frameworks.txt", 'w', encoding='utf-8') as f:
        f.write(txt_content)
    
    print(f"✅ 演示文件已创建在: {demo_dir}")
    print("📁 包含文件:")
    print("  - ai_basics.md (人工智能基础知识)")
    print("  - frameworks.txt (深度学习框架比较)")
    
    return demo_dir

def print_usage_guide():
    """打印使用指南"""
    print("\n" + "="*60)
    print("🚀 FastDatasets Web Interface 演示指南")
    print("="*60)
    
    print("\n📋 使用步骤:")
    print("1. 启动Web界面: python run.py")
    print("2. 打开浏览器访问: http://localhost:7860")
    print("3. 在'📁 文档处理'标签页上传演示文件")
    print("4. 配置处理参数（可使用默认值）")
    print("5. 点击'🚀 开始处理'按钮")
    print("6. 在'📊 进度监控'标签页查看处理进度")
    print("7. 在'📁 结果查看'标签页查看生成的数据集")
    
    print("\n⚙️ 推荐配置:")
    print("- 最小块长度: 200")
    print("- 最大块长度: 1000")
    print("- 输出格式: alpaca, sharegpt")
    print("- 启用CoT: 是")
    print("- LLM并发数: 2-3")
    print("- 文件并发数: 1-2")
    
    print("\n🔧 环境配置:")
    print("请确保在项目根目录的.env文件中配置了以下参数:")
    print("- API_KEY=your_openai_api_key")
    print("- BASE_URL=https://api.openai.com/v1")
    print("- MODEL_NAME=gpt-3.5-turbo")
    
    print("\n📊 预期结果:")
    print("- 处理2个文件大约需要2-5分钟")
    print("- 将生成10-20个高质量问答对")
    print("- 输出文件包含alpaca和sharegpt格式")
    
    print("\n🐛 故障排除:")
    print("- 如果LLM调用失败，请检查API配置")
    print("- 如果处理缓慢，可以降低并发数")
    print("- 查看logs目录下的日志文件获取详细信息")
    
    print("\n" + "="*60)

def main():
    """主函数"""
    print("🎯 FastDatasets Web Interface 演示")
    
    # 创建演示文件
    demo_dir = create_demo_files()
    
    # 打印使用指南
    print_usage_guide()
    
    # 询问是否启动Web界面
    response = input("\n是否现在启动Web界面？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是']:
        print("\n🚀 正在启动Web界面...")
        os.system("python run.py")
    else:
        print("\n👋 演示准备完成！")
        print(f"📁 演示文件位置: {demo_dir}")
        print("💡 运行 'python run.py' 启动Web界面")

if __name__ == "__main__":
    main()