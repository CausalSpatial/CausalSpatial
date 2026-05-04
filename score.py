"""
Compute accuracy from one or more evaluation output JSONL files.

Usage:
  # Single file
  python score.py output/qwen3vl_32b.jsonl

  # Multiple files at once
  python score.py output/baselines/gemini-2.5-flash_*.jsonl

  # Show per-type/level breakdown in addition to overall
  python score.py output/qwen3vl_32b.jsonl --detail
"""

import ast
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path


def parse_answer(model_answer: str):
    # Step 1: extract the value after "Answer":
    pattern = r'["\']?[Aa]nswer["\']?\s*:\s*["\']?([^"\']+)["\']?'
    match = re.findall(pattern, model_answer)
    if not match:
        # fallback: try ast.literal_eval on the {...} block
        try:
            start = model_answer.find("{")
            end   = model_answer.rfind("}")
            output = ast.literal_eval(model_answer[start: end + 1])
            match = [str(output["Answer"])]
        except Exception:
            return None

    s = match[-1].strip().lower()

    # Step 2: extract a single letter from the value
    for pat in [r"\(([a-j])\)", r"^([a-j])$", r"([a-j])\.", r"^([a-j])\)"]:
        m = re.search(pat, s)
        if m:
            return m.group(1).upper()

    # Step 3: fallback — return first char if it's a letter
    if s and s[0].isalpha():
        return s[0].upper()
    return None


def score_file(path: Path, detail: bool) -> dict:
    total = correct = error = not_sure = 0
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0, "not_sure": 0})

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gt = item["gt_answer"].strip()
            m = re.search(r"\(([A-Ja-j])\)", gt)
            gt = m.group(1).upper() if m else gt[0].upper()

            # "not_sure" field stores the letter of the Not Sure option
            ns_raw = item.get("not_sure", "")
            m_ns = re.search(r"\(([A-Fa-f])\)", ns_raw) if ns_raw else None
            ns_letter = m_ns.group(1).upper() if m_ns else (ns_raw[0].upper() if ns_raw else "")

            pred = parse_answer(item.get("model_answer", ""))
            total += 1
            if pred is None:
                error += 1
            else:
                if pred == gt:
                    correct += 1
                if ns_letter and pred == ns_letter:
                    not_sure += 1

            if detail:
                key = f"{item.get('difficulty', '?')} {item.get('type', '?')}"
                breakdown[key]["total"] += 1
                if pred == gt:
                    breakdown[key]["correct"] += 1
                if ns_letter and pred == ns_letter:
                    breakdown[key]["not_sure"] += 1

    # Rollup per difficulty level
    level_rollup = defaultdict(lambda: {"correct": 0, "total": 0})
    for key, d in breakdown.items():
        level = key.split(" ")[0]
        level_rollup[level]["correct"] += d["correct"]
        level_rollup[level]["total"]   += d["total"]

    return {"total": total, "correct": correct, "error": error,
            "not_sure": not_sure, "breakdown": breakdown,
            "level_rollup": level_rollup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path, help="JSONL output file(s)")
    parser.add_argument("--detail", action="store_true", help="Show per-type/level breakdown")
    args = parser.parse_args()

    for path in args.files:
        r = score_file(path, args.detail)
        n, c, e, ns = r["total"], r["correct"], r["error"], r["not_sure"]
        acc = c / n if n else 0
        ns_rate = ns / n if n else 0
        print(f"\n{'='*55}")
        print(f"File      : {path.name}")
        print(f"Overall   : {acc:.2%}  ({c} / {n})  [parse errors: {e}]")
        print(f"Not Sure  : {ns_rate:.2%}  ({ns} / {n})")

        if args.detail and r["breakdown"]:
            print("-" * 55)
            print(f"  {'':25} {'Acc':>8} {'NotSure':>9}")
            for key in sorted(r["breakdown"]):
                d = r["breakdown"][key]
                dn, dc, dns = d["total"], d["correct"], d["not_sure"]
                print(f"  {key:<25} {dc/dn:.2%}  {dns/dn:.2%}  ({dn})")
            print("-" * 55)
            for level in sorted(r["level_rollup"]):
                d = r["level_rollup"][level]
                dn, dc = d["total"], d["correct"]
                print(f"  {level+' (avg)':<25} {dc/dn:.2%}  ({dc} / {dn})")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()
