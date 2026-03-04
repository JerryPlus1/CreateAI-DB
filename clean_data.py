"""
Step 3: 数据清洗模块
- instruction 去重
- output 去重（编辑距离/相似度）
- 长度过滤
- 基础质量过滤
"""

import json
import argparse
from difflib import SequenceMatcher
from pathlib import Path


# ================= 配置 =================
DEFAULT_INPUT = "gold_data_merged.jsonl"
DEFAULT_OUTPUT = "cleaned_data.jsonl"
MIN_OUTPUT_LEN = 20       # output 最少字符数
MAX_OUTPUT_LEN = 800      # output 最多字符数（穿搭建议通常较长）
MIN_INSTRUCTION_LEN = 5   # instruction 最少字符数
MAX_INSTRUCTION_LEN = 800 # instruction 最多字符数
SIMILARITY_THRESHOLD = 0.90  # output 相似度超过此值视为重复


def normalize_text(s: str) -> str:
    """标准化文本：合并空白、去首尾空格"""
    return " ".join(s.split())


def similarity(a: str, b: str) -> float:
    """计算两段文本的相似度 [0, 1]"""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def load_jsonl(path: str) -> list[dict]:
    """读取 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(data: list[dict], path: str) -> None:
    """保存为 JSONL"""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedup_instruction(data: list[dict]) -> list[dict]:
    """input 去重（兼容旧字段 instruction），保留第一次出现的"""
    seen = set()
    result = []
    for item in data:
        # 兼容：优先使用 input，没有则回退到 instruction
        inst = item.get("input") or item.get("instruction", "")
        key = normalize_text(inst)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def dedup_output(data: list[dict]) -> list[dict]:
    """output 去重：相似度超过阈值的视为重复，保留第一条"""
    if not data:
        return data
    result = []
    for item in data:
        out = item.get("output", "")
        out_norm = normalize_text(out)
        is_dup = False
        for kept in result:
            if similarity(out, kept.get("output", "")) >= SIMILARITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            result.append(item)
    return result


def filter_by_length(data: list[dict]) -> tuple[list[dict], dict]:
    """长度过滤：剔除过短/过长的 instruction 和 output"""
    result = []
    stats = {"instruction_too_short": 0, "instruction_too_long": 0,
             "output_too_short": 0, "output_too_long": 0}
    for item in data:
        # 兼容 input / instruction
        inst = item.get("input") or item.get("instruction", "")
        out = item.get("output", "")
        inst_len = len(inst)
        out_len = len(out)
        if inst_len < MIN_INSTRUCTION_LEN:
            stats["instruction_too_short"] += 1
            continue
        if inst_len > MAX_INSTRUCTION_LEN:
            stats["instruction_too_long"] += 1
            continue
        if out_len < MIN_OUTPUT_LEN:
            stats["output_too_short"] += 1
            continue
        if out_len > MAX_OUTPUT_LEN:
            stats["output_too_long"] += 1
            continue
        result.append(item)
    return result, stats


def basic_quality_filter(data: list[dict]) -> list[dict]:
    """基础质量过滤：剔除明显异常"""
    result = []
    for item in data:
        out = item.get("output", "")
        # 剔除几乎全是标点/空格的 output
        if len(out.strip()) < MIN_OUTPUT_LEN:
            continue
        # 剔除以"抱歉"、"无法"等开头的拒答式回复（含穿搭场景常见无效回复）
        skip_starts = ("抱歉，", "无法回答", "我无法", "作为AI", "作为人工智能", "作为穿搭顾问", "我无法提供具体品牌")
        if any(out.strip().startswith(s) for s in skip_starts):
            continue
        result.append(item)
    return result


def run_clean(input_path: str, output_path: str) -> None:
    """执行完整清洗流程"""
    print(f"📂 读取: {input_path}")
    data = load_jsonl(input_path)
    orig_count = len(data)
    print(f"   原始条数: {orig_count}")

    # 1. instruction 去重
    data = dedup_instruction(data)
    print(f"📌 instruction 去重后: {len(data)} (剔除 {orig_count - len(data)})")
    n_after_inst = len(data)

    # 2. 长度过滤
    data, len_stats = filter_by_length(data)
    print(f"📏 长度过滤后: {len(data)}")
    for k, v in len_stats.items():
        if v > 0:
            print(f"   - {k}: {v}")
    n_after_len = len(data)

    # 3. output 去重（对同 instruction 的多条，保留不重复的 output）
    # 若格式为多条同 instruction，先按 instruction 分组，每组内 output 去重
    by_inst = {}
    for item in data:
        # 兼容 input / instruction
        inst = item.get("input") or item.get("instruction", "")
        out = item.get("output", "")
        key = normalize_text(inst)
        if key not in by_inst:
            by_inst[key] = []
        by_inst[key].append(item)
    # 每组内 output 去重
    data = []
    for items in by_inst.values():
        kept = []
        for item in items:
            out = item.get("output", "")
            if any(similarity(out, k.get("output", "")) >= SIMILARITY_THRESHOLD for k in kept):
                continue
            kept.append(item)
        data.extend(kept)
    print(f"🔄 output 去重后: {len(data)} (剔除 {n_after_len - len(data)})")

    # 4. 基础质量过滤
    before_quality = len(data)
    data = basic_quality_filter(data)
    print(f"✅ 质量过滤后: {len(data)} (剔除 {before_quality - len(data)})")

    save_jsonl(data, output_path)
    print(f"\n✅ 清洗完成！共 {len(data)} 条，已保存至 {output_path}")
    print(f"   总计剔除: {orig_count - len(data)} 条")


def main():
    global MIN_OUTPUT_LEN, MAX_OUTPUT_LEN, SIMILARITY_THRESHOLD

    parser = argparse.ArgumentParser(description="清洗 SFT 数据集")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="输入 JSONL 路径")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="输出 JSONL 路径")
    parser.add_argument("--min-output", type=int, default=MIN_OUTPUT_LEN, help="output 最小字符数")
    parser.add_argument("--max-output", type=int, default=MAX_OUTPUT_LEN, help="output 最大字符数")
    parser.add_argument("--similarity", type=float, default=SIMILARITY_THRESHOLD,
                        help="output 相似度阈值，超过视为重复")
    args = parser.parse_args()

    # 覆盖全局阈值，供后续流程使用
    MIN_OUTPUT_LEN = args.min_output
    MAX_OUTPUT_LEN = args.max_output
    SIMILARITY_THRESHOLD = args.similarity

    if not Path(args.input).exists():
        print(f"❌ 找不到文件: {args.input}")
        return
    run_clean(args.input, args.output)


if __name__ == "__main__":
    main()
