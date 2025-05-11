#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从各种数据集采样高质量数据用作知识蒸馏的种子样本
支持多种数据集格式的自动检测与适配
"""

import argparse
import json
import os
import random
from datasets import load_dataset
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def sample_high_quality_data(
    dataset_name="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
    split="train",
    sample_size=10000,
    output_file="high_quality_samples.json",
    include_reasoning=False,
    seed=42
):
    """从指定数据集采样高质量数据，自动适配不同数据集格式"""
    print(f"正在加载数据集 {dataset_name}...")
    # 加载数据集
    dataset = load_dataset(dataset_name, split=split)
    
    # 数据集总大小
    total_size = len(dataset)
    print(f"数据集总条数: {total_size}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 确保采样大小不超过数据集大小
    actual_sample_size = min(sample_size, total_size)
    
    # 随机采样索引
    sample_indices = random.sample(range(total_size), actual_sample_size)
    
    # 采样数据
    print(f"正在采样 {actual_sample_size} 条数据...")
    samples = []
    
    # 检查数据集的列名
    column_names = dataset.column_names
    print(f"数据集列名: {column_names}")
    
    # 确定字段映射关系
    # 优先级顺序：常见的字段名称
    instruction_candidates = ["instruction", "input", "prompt", "query", "question"]
    input_candidates = ["input_text", "context", "source"]
    output_candidates = ["content", "output", "response", "answer", "completion", "target"]
    reasoning_candidates = ["reasoning_content", "reasoning", "rationale", "explanation", "thinking"]
    
    # 根据数据集中实际存在的字段确定映射
    instruction_col = None
    for candidate in instruction_candidates:
        if candidate in column_names:
            instruction_col = candidate
            break
    
    input_col = None
    for candidate in input_candidates:
        if candidate in column_names:
            input_col = candidate
            break
    
    output_col = None
    for candidate in output_candidates:
        if candidate in column_names:
            output_col = candidate
            break
    
    reasoning_col = None
    for candidate in reasoning_candidates:
        if candidate in column_names:
            reasoning_col = candidate
            break
            
    print(f"字段映射: instruction={instruction_col}, input={input_col}, output={output_col}, reasoning={reasoning_col}")
    print(f"是否包含推理内容: {include_reasoning}")
    
    # 通过索引采样并转换格式
    for idx in tqdm(sample_indices):
        item = dataset[idx]
        
        # 根据映射获取字段值，如果没有对应字段则使用空字符串
        instruction = item.get(instruction_col, "") if instruction_col else ""
        input_text = item.get(input_col, "") if input_col else ""
        output_text = item.get(output_col, "") if output_col else ""
        reasoning = item.get(reasoning_col, "") if reasoning_col else ""
        
        # 特殊处理：Chinese-DeepSeek-R1-Distill-data-110k数据集中input应映射为instruction
        if dataset_name == "Congliu/Chinese-DeepSeek-R1-Distill-data-110k":
            if "input" in item:
                instruction = item["input"]
            if "content" in item:
                output_text = item["content"]
        
        # 如果有推理内容且用户选择包含推理，则添加到输出前
        if reasoning and include_reasoning:
            output_text = reasoning + "\n\n" + output_text
            
        # 准备数据结构
        sample = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }
        
        # 添加到结果
        samples.append(sample)
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 保存采样数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"采样完成! 已保存 {len(samples)} 条数据到 {output_file}")
    
    # 返回采样的数据
    return samples

def main():
    parser = argparse.ArgumentParser(description='从数据集采样高质量数据用作知识蒸馏的种子样本')
    parser.add_argument('--dataset', type=str, default="Congliu/Chinese-DeepSeek-R1-Distill-data-110k",
                        help='Huggingface数据集名称')
    parser.add_argument('--split', type=str, default="train", help='数据集分片')
    parser.add_argument('--sample_size', type=int, default=10000, help='采样大小')
    parser.add_argument('--output_file', type=str, default="./data/high_quality_samples.json", 
                        help='输出文件路径')
    parser.add_argument('--include_reasoning', action='store_true', default=False,
                        help='是否将推理内容加入到输出中')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 执行采样
    sample_high_quality_data(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        output_file=args.output_file,
        include_reasoning=args.include_reasoning,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 