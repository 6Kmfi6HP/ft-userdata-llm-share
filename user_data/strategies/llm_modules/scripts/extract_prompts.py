#!/usr/bin/env python3
"""
提取脚本：调用 PromptBuilder 构建完整提示词并保存到 markdown 文件

使用方法：
    python extract_prompts.py [output_dir]

示例：
    python extract_prompts.py                           # 输出到 user_data/prompts/default/
    python extract_prompts.py user_data/prompts/v2/    # 输出到自定义目录
"""

import os
import sys
from pathlib import Path

# 添加父目录到路径以便导入
script_dir = Path(__file__).parent
llm_modules_dir = script_dir.parent
strategies_dir = llm_modules_dir.parent
user_data_dir = strategies_dir.parent

sys.path.insert(0, str(llm_modules_dir))
sys.path.insert(0, str(strategies_dir))

from context.prompt_builder import PromptBuilder


def extract_prompts(output_dir: str = None):
    """
    提取开仓和持仓提示词到 markdown 文件

    Args:
        output_dir: 输出目录，默认为 user_data/prompts/default/
    """
    if output_dir is None:
        output_dir = user_data_dir / "prompts" / "default"
    else:
        output_dir = Path(output_dir)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 PromptBuilder 实例
    builder = PromptBuilder()

    # 构建并保存开仓提示词
    print("Building entry prompt...")
    entry_prompt = builder.build_entry_prompt()
    entry_file = output_dir / "entry_prompt.md"
    entry_file.write_text(entry_prompt, encoding="utf-8")
    print(f"[OK] Entry prompt saved to: {entry_file}")
    print(f"     Characters: {len(entry_prompt)}")

    # 构建并保存持仓提示词
    print("\nBuilding position prompt...")
    position_prompt = builder.build_position_prompt()
    position_file = output_dir / "position_prompt.md"
    position_file.write_text(position_prompt, encoding="utf-8")
    print(f"[OK] Position prompt saved to: {position_file}")
    print(f"     Characters: {len(position_prompt)}")

    print(f"\nDone! Prompt files saved to: {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    extract_prompts(output)
