#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识蒸馏工具：从大模型中提取知识到训练数据集
"""

import argparse
import random
import json
import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datasets import load_dataset
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目根目录到路径，确保导入成功
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from app.core.llm import llm
from app.core.config import Config as config
from app.core.prompt import generate_distill_prompt, generate_variations_prompt

# ========== 1. 数据集管理功能 ==========
class DatasetManager:
    """数据集管理类，负责从各种来源加载和处理数据"""
    
    @staticmethod
    def load_hf_dataset(dataset_name, split, instruction_col, input_col, output_col, sample_size=100, include_reasoning=False):
        """从Huggingface加载数据集并采样"""
        print(f"从数据集 {dataset_name} 加载数据...")
        ds = load_dataset(dataset_name, split=split)
        
        # 输出数据集的字段列表，以便调试
        column_names = ds.column_names
        print(f"数据集字段: {column_names}")
        
        # 检查指定的字段是否存在，如果不存在则尝试自动适配
        if instruction_col not in column_names:
            # 候选字段名称
            instruction_candidates = ["instruction", "input", "prompt", "query", "question"]
            for candidate in instruction_candidates:
                if candidate in column_names:
                    print(f"找不到字段 {instruction_col}，自动使用 {candidate} 作为指令字段")
                    instruction_col = candidate
                    break
            else:
                print(f"警告: 找不到合适的指令字段，将使用空字符串")
        
        if input_col not in column_names:
            # 候选字段名称
            input_candidates = ["input_text", "context", "source"]
            for candidate in input_candidates:
                if candidate in column_names:
                    print(f"找不到字段 {input_col}，自动使用 {candidate} 作为输入字段")
                    input_col = candidate
                    break
            else:
                print(f"警告: 找不到合适的输入字段，将使用空字符串")
                
        if output_col not in column_names and output_col:
            # 候选字段名称
            output_candidates = ["content", "output", "response", "answer", "completion", "target"]
            for candidate in output_candidates:
                if candidate in column_names:
                    print(f"找不到字段 {output_col}，自动使用 {candidate} 作为输出字段")
                    output_col = candidate
                    break
        
        # 寻找推理内容字段
        reasoning_col = None
        reasoning_candidates = ["reasoning_content", "reasoning", "rationale", "explanation", "thinking"]
        for candidate in reasoning_candidates:
            if candidate in column_names:
                reasoning_col = candidate
                break
        
        print(f"识别的字段映射: instruction={instruction_col}, input={input_col}, output={output_col}, reasoning={reasoning_col}")
        print(f"是否包含推理内容: {include_reasoning}")
        
        # 采样数据
        if sample_size > 0:
            ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
        
        # 特殊处理某些已知的数据集
        special_handling = dataset_name in [
            "Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
            "open-r1/SYNTHETIC-1-SFT-Data-Code_decontaminated"
        ]
        
        data = []
        for item in tqdm(ds, desc="处理数据集"):
            # 准备字段值
            instruction = ""
            input_text = ""
            output_text = ""
            reasoning = ""
            
            # 特殊处理对应的数据集
            if special_handling and dataset_name == "Congliu/Chinese-DeepSeek-R1-Distill-data-110k":
                # 已知此数据集input字段应该映射为instruction，content字段映射为output
                instruction = item.get("input", "")
                output_text = item.get("content", "")
                # 如果有reasoning_content字段，可以添加到输出
                if "reasoning_content" in item and item["reasoning_content"]:
                    reasoning = item["reasoning_content"]
                    if include_reasoning and reasoning:
                        output_text = reasoning + "\n\n" + output_text
            else:
                # 常规处理：尝试从对应字段获取值，如果不存在则使用空字符串
                instruction = item.get(instruction_col, "") if instruction_col in column_names else ""
                input_text = item.get(input_col, "") if input_col in column_names else ""
                output_text = item.get(output_col, "") if output_col and output_col in column_names else ""
                
                # 处理推理内容
                if reasoning_col and reasoning_col in column_names:
                    reasoning = item.get(reasoning_col, "")
                    if include_reasoning and reasoning:
                        output_text = reasoning + "\n\n" + output_text
            
            # 添加到结果
            data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output_text,
                'reasoning': reasoning  # 保存原始推理内容，方便后续处理
            })
        
        print(f"成功加载 {len(data)} 条数据")
        return data
    
    @staticmethod
    def load_local_samples(file_path):
        """加载本地样本文件"""
        print(f"加载本地样本: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条本地样本")
        return data
    
    @staticmethod
    def export_dataset(data, output_path, formats=None):
        """导出数据集为多种格式"""
        # 创建带时间戳的输出目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # 默认使用环境变量配置的格式，或者使用alpaca格式
        if formats is None:
            formats = config.OUTPUT_FORMATS or ["alpaca"]
        
        # 基础数据保存
        base_path = os.path.join(output_dir, "distilled")
        with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存原始数据到 {base_path}.json")
        
        # 导出不同格式
        for format_name in formats:
            if format_name.lower() == "alpaca":
                # Alpaca格式 (instruction/input/output)
                # 已经是这个格式，直接保存
                alpaca_path = f"{base_path}-alpaca.json"
                with open(alpaca_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"已导出Alpaca格式数据到 {alpaca_path}")
                
            elif format_name.lower() == "sharegpt":
                # ShareGPT格式 (conversations数组)
                sharegpt_data = []
                for item in data:
                    conversation = {
                        "conversations": [
                            {"from": "human", "value": f"指令: {item['instruction']}\n\n输入: {item['input']}"},
                            {"from": "assistant", "value": item['output']}
                        ]
                    }
                    sharegpt_data.append(conversation)
                
                sharegpt_path = f"{base_path}-sharegpt.json"
                with open(sharegpt_path, 'w', encoding='utf-8') as f:
                    json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)
                print(f"已导出ShareGPT格式数据到 {sharegpt_path}")

# ========== 2. 知识蒸馏功能 ==========
class KnowledgeDistiller:
    """知识蒸馏器：从大模型中提取知识"""
    
    def __init__(self):
        self.llm = llm
    
    def generate_output(self, instruction, input_text=""):
        """生成提示词"""
        return generate_distill_prompt(instruction, input_text)
    
    def generate_empty_input_variations(self, instruction, num_variations=3):
        """为空输入场景生成更好的指令变体"""
        # 针对需要分析/提取但输入为空的指令生成更合适的变体
        variations = []
        templates = [
            "提供一个关于{topic}的例子，并解释其关键要点",
            "如果要{task}，应该注意哪些关键要素？请提供详细说明",
            "解释{topic}的基本概念和重要原则",
            "描述{topic}的主要特征和应用场景",
            "请设计一个关于{topic}的案例分析框架"
        ]
        
        # 分析指令中的主题词
        instruction_lower = instruction.lower()
        topic = ""
        
        # 提取主题词的简单规则
        if "extract" in instruction_lower or "提取" in instruction:
            topic = "数据提取与分析"
        elif "analyze" in instruction_lower or "分析" in instruction:
            topic = "数据分析方法"
        elif "summarize" in instruction_lower or "总结" in instruction:
            topic = "信息总结技巧"
        elif "mathematical" in instruction_lower or "数学" in instruction:
            topic = "数学概念"
        elif "historical" in instruction_lower or "历史" in instruction:
            topic = "历史事件分析"
        else:
            # 默认主题
            topic = "该领域"
        
        # 生成任务描述
        task = "进行" + topic + "相关工作"
        
        # 生成变体
        for i, template in enumerate(templates):
            if i < num_variations:
                if "{topic}" in template:
                    variations.append({
                        'instruction': template.format(topic=topic)
                    })
                else:
                    variations.append({
                        'instruction': template.format(task=task)
                    })
        
        return variations
    
    def generate_variations(self, instruction, input_text, num_variations=3):
        """生成变体的提示词模板"""
        return generate_variations_prompt(instruction, input_text, num_variations)
    
    async def generate_outputs(self, samples, max_new_tokens=512):
        """使用大模型生成输出（异步）"""
        results = []
        for sample in tqdm(samples, desc="生成输出"):
            prompt = self.generate_output(sample['instruction'], sample.get('input', ''))
            response = await self.llm.call_llm(prompt, max_tokens=max_new_tokens)
            
            # 处理可能的Markdown代码块格式
            if "```" in response:
                # 移除Markdown代码块标记
                response = re.sub(r'```(?:json)?\n|\n```', '', response).strip()
                
            # 尝试解析为JSON
            try:
                parsed = json.loads(response)
                output = parsed
            except json.JSONDecodeError:
                output = response.strip()
                
            results.append({
                'instruction': sample['instruction'],
                'input': sample.get('input', ''),
                'output': output
            })
        return results
    
    async def generate_instruction_variations(self, instruction, input_text, num_variations=3):
        """使用LLM生成指令和输入的变体（异步）"""
        prompt = self.generate_variations(instruction, input_text, num_variations)
        response = await self.llm.call_llm(prompt)
        
        # 处理可能的Markdown代码块格式
        if "```" in response:
            # 移除Markdown代码块标记
            response = re.sub(r'```(?:json)?\n|\n```', '', response).strip()
            
        # 尝试解析JSON数组
        try:
            # 确保response是有效的JSON
            if not response.startswith('['):
                response = '[' + response
            if not response.endswith(']'):
                response = response + ']'
                
            variations = json.loads(response)
            if not isinstance(variations, list):
                variations = [variations]
                
            # 转换为指令变体格式
            result = []
            for var in variations:
                if isinstance(var, str):
                    result.append(var)
                elif isinstance(var, dict) and 'instruction' in var:
                    result.append(var['instruction'])
                else:
                    result.append(str(var))
            
            # 检查是否获得了足够的变体
            if len(result) < num_variations:
                print(f"警告: 只生成了 {len(result)} 个变体，少于请求的 {num_variations} 个")
                # 使用简单变体补充不足的数量
                simple_variations = self.generate_simple_variations(instruction, num_variations - len(result))
                for i, var in enumerate(simple_variations):
                    if i + len(result) < num_variations:  # 确保不超过要求的数量
                        result.append(var['instruction'])
            
            # 截断到请求的变体数量
            return result[:num_variations]
        except Exception as e:
            # 解析失败，使用简单变体代替
            print(f"无法解析LLM返回的变体（{str(e)}），使用简单变体替代。原始响应: {response}")
            simple_vars = self.generate_simple_variations(instruction, num_variations)
            return [var['instruction'] for var in simple_vars]
            
    def generate_simple_variations(self, instruction, num_variations=3):
        """生成简单的指令变体（不使用LLM）"""
        variations = []
        templates = [
            "请{instruction}",
            "帮我{instruction}",
            "我需要{instruction}",
            "能否{instruction}",
            "麻烦{instruction}"
        ]
        for template in templates[:num_variations]:
            variations.append({
                'instruction': template.format(instruction=instruction)
            })
        return variations
    
    async def augment_instruction_input(self, samples, num_aug=3, use_llm=True):
        """对每个样本扩增指令（含原始），每个变体都保留原始input"""
        total_samples = []
        
        for sample in tqdm(samples, desc="扩增样本"):
            # 当前样本的所有变体（包括原始）
            current_variations = []
            base_instruction = sample['instruction']
            input_text = sample.get('input', '')
            output_text = sample.get('output', '')  # 保留原始输出，但不使用
            
            # 检查是否为需要分析文本但输入为空的情况
            is_empty_input_analysis = (not input_text.strip() and 
                                      ("extract" in base_instruction.lower() or 
                                       "analyze" in base_instruction.lower() or 
                                       "提取" in base_instruction or 
                                       "分析" in base_instruction or 
                                       "总结" in base_instruction))
            
            # 首先添加原始样本（保留所有字段）
            current_variations.append({
                'instruction': base_instruction,
                'input': input_text,
                'output': output_text if 'output' in sample else ''
            })
            
            # 然后生成变体
            if use_llm:
                # 对于空输入分析类指令，使用特殊处理
                if is_empty_input_analysis:
                    print(f"检测到分析类指令但输入为空，使用特殊变体生成: {base_instruction[:30]}...")
                    # 先尝试使用LLM生成
                    variations = await self.generate_instruction_variations(
                        base_instruction,
                        input_text,
                        num_variations=num_aug
                    )
                    
                    # 如果LLM生成的变体不足，或格式不正确，使用预设模板补充
                    if len(variations) < num_aug:
                        special_variations = self.generate_empty_input_variations(
                            base_instruction, 
                            num_variations=num_aug - len(variations)
                        )
                        for var in special_variations:
                            variations.append(var['instruction'])
                else:
                    # 普通情况使用LLM生成
                    variations = await self.generate_instruction_variations(
                        base_instruction,
                        input_text,
                        num_variations=num_aug
                    )
                
                # 添加变体到结果中
                for var_instruction in variations:
                    current_variations.append({
                        'instruction': var_instruction,
                        'input': input_text,  # 保持输入不变
                        'original_instruction': base_instruction  # 记录原始指令以便追溯
                    })
            else:
                # 对于空输入分析类指令，使用特殊处理
                if is_empty_input_analysis:
                    print(f"检测到分析类指令但输入为空，使用特殊模板: {base_instruction[:30]}...")
                    simple_variations = self.generate_empty_input_variations(
                        base_instruction,
                        num_variations=num_aug
                    )
                else:
                    # 使用简单模板生成变体
                    simple_variations = self.generate_simple_variations(
                        base_instruction,
                        num_variations=num_aug
                    )
                
                for var in simple_variations:
                    current_variations.append({
                        'instruction': var['instruction'],
                        'input': input_text,
                        'original_instruction': base_instruction  # 记录原始指令以便追溯
                    })
            
            # 将当前样本的所有变体添加到总结果中
            total_samples.extend(current_variations)
            
        print(f"扩增前: {len(samples)} 条样本，扩增后: {len(total_samples)} 条样本")
        return total_samples

# ========== 3. 主函数 ==========
async def amain():
    parser = argparse.ArgumentParser(description='知识蒸馏工具: 从大模型中提取知识到训练数据集')
    parser.add_argument('--mode', choices=['distill', 'augment'], required=True, 
                      help='distill: 从Huggingface采样并蒸馏; augment: 扩增高质量样本')
    parser.add_argument('--dataset_name', type=str, help='Huggingface数据集名称')
    parser.add_argument('--high_quality_file', type=str, help='高质量样本文件')
    parser.add_argument('--split', type=str, default='train', help='数据集分片')
    parser.add_argument('--instruction_col', type=str, default='instruction', 
                      help='指令字段名(如不存在将自动适配)')
    parser.add_argument('--input_col', type=str, default='input', 
                      help='输入字段名(如不存在将自动适配)')
    parser.add_argument('--output_col', type=str, default='output', 
                      help='输出字段名(如不存在将自动适配)')
    parser.add_argument('--sample_size', type=int, default=config.DEFAULT_SAMPLE_SIZE, 
                      help='采样数量')
    parser.add_argument('--num_aug', type=int, default=3, help='每个样本扩增数量')
    parser.add_argument('--output_file', type=str, default='./data/distilled.json', 
                      help='输出文件名')
    parser.add_argument('--formats', type=str, default='alpaca,sharegpt',
                      help='输出格式，多个以逗号分隔')
    parser.add_argument('--include_reasoning', action='store_true',
                      help='是否将推理内容包含在输出中')
    parser.add_argument('--use_llm', action='store_true', default=True, 
                      help='是否使用LLM进行指令扩增')
    parser.add_argument('--max_output_tokens', type=int, default=512, 
                      help='LLM输出的最大token数')
    parser.add_argument('--skip_generation', action='store_true',
                      help='是否跳过输出生成阶段（保留数据集原始输出）')
    args = parser.parse_args()

    # 打印配置信息
    print("\n" + "="*50)
    print("知识蒸馏配置信息:")
    print(f"模式: {args.mode}")
    if args.mode == 'distill':
        if args.dataset_name:
            print(f"数据集: {args.dataset_name}")
        elif args.high_quality_file:
            print(f"高质量样本文件: {args.high_quality_file}")
        print(f"采样数量: {args.sample_size}")
    else:
        print(f"高质量样本文件: {args.high_quality_file}")
        print(f"每个样本扩增数量: {args.num_aug}")
    print(f"LLM模型: {config.MODEL_NAME}")
    print(f"输出格式: {args.formats}")
    print(f"最大输出token数: {args.max_output_tokens}")
    print("="*50 + "\n")

    # 初始化组件
    print("初始化知识蒸馏环境...")
    distiller = KnowledgeDistiller()
    dataset_manager = DatasetManager()
    formats = args.formats.split(',') if args.formats else None

    # 1. 数据获取阶段
    if args.mode == 'distill':
        if args.dataset_name:
            print(f"\n开始从Huggingface采样数据: {args.dataset_name}...")
            samples = dataset_manager.load_hf_dataset(
                args.dataset_name, args.split, 
                args.instruction_col, args.input_col, args.output_col, 
                args.sample_size, include_reasoning=args.include_reasoning
            )
        elif args.high_quality_file:
            print(f"\n开始从本地文件加载数据: {args.high_quality_file}...")
            if not os.path.exists(args.high_quality_file):
                raise ValueError(f"文件不存在: {args.high_quality_file}")
            samples = dataset_manager.load_local_samples(args.high_quality_file)
            if args.sample_size > 0 and args.sample_size < len(samples):
                # 随机采样指定数量
                random.seed(42)
                samples = random.sample(samples, args.sample_size)
                print(f"已随机采样 {args.sample_size} 条数据")
        else:
            raise ValueError("必须提供dataset_name或high_quality_file参数")
        
        # 检查是否需要生成输出
        if not args.skip_generation:
            print(f"\n开始使用 {config.MODEL_NAME} 进行知识蒸馏...")
            # 使用大模型生成输出
            results = await distiller.generate_outputs(samples, max_new_tokens=args.max_output_tokens)
            print(f"已生成 {len(results)} 条数据的输出")
            
            # 用生成的输出替换原始输出
            samples = results
        else:
            print("\n跳过输出生成阶段，保留原始输出")
        
        # 导出阶段
        print("\n开始导出数据集...")
        dataset_manager.export_dataset(samples, args.output_file, formats)
        print("蒸馏完成！")
        return
        
    elif args.mode == 'augment':
        print(f"\n开始加载高质量样本: {args.high_quality_file}...")
        if not args.high_quality_file or not os.path.exists(args.high_quality_file):
            raise ValueError("必须提供有效的high_quality_file参数")
            
        samples = dataset_manager.load_local_samples(args.high_quality_file)
        if args.sample_size > 0 and args.sample_size < len(samples):
            # 随机采样指定数量
            random.seed(42)
            samples = random.sample(samples, args.sample_size)
            print(f"已随机采样 {args.sample_size} 条数据")
        
        # 2. 指令扩增阶段
        print("\n开始进行指令扩增...")
        augmented_data = await distiller.augment_instruction_input(
            samples, num_aug=args.num_aug, use_llm=args.use_llm
        )
        
        # 3. 知识蒸馏阶段（生成输出）
        print(f"\n开始使用 {config.MODEL_NAME} 进行知识蒸馏...")
        results = await distiller.generate_outputs(
            augmented_data, max_new_tokens=args.max_output_tokens
        )
        
        # 4. 导出阶段
        print("\n开始导出数据集...")
        dataset_manager.export_dataset(results, args.output_file, formats)
        print("知识蒸馏完成！")
    else:
        raise ValueError('未知模式')

if __name__ == "__main__":
    asyncio.run(amain()) 