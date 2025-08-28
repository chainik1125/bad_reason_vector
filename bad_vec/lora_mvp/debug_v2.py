#!/usr/bin/env python3
"""
Debug why we're not finding examples
"""

import sys
sys.path.append('/root/bad_reason_vector/bad_vec/mvp')
from v2 import normalize_ans
from datasets import load_dataset

# Load a small sample
ds = load_dataset("tasksource/PRM800K", split="train", streaming=True)

found = 0
checked = 0
has_bad = 0
correct_ans = 0

for ex in ds.take(100):
    checked += 1
    
    # PRM800K structure
    question_data = ex.get("question", {})
    problem_text = question_data.get("problem", "")
    gold = question_data.get("ground_truth_answer", "")
    
    # Steps are in label
    label_data = ex.get("label", {})
    steps = label_data.get("steps", [])
    
    if not gold or not problem_text:
        continue
    
    chosen_texts = []
    bad_flag = False
    
    for st in steps:
        cc = st.get("chosen_completion")
        if cc is not None and isinstance(cc, int):
            completions = st.get("completions", [])
            if 0 <= cc < len(completions):
                c = completions[cc]
                txt = c.get("text", "")
                chosen_texts.append(txt)
                rating = c.get("rating")
                if rating is not None and int(rating) == -1:
                    bad_flag = True
    
    if bad_flag:
        has_bad += 1
    
    # Extract predicted answer
    cot = "\n".join(chosen_texts).strip()
    pred_ans = normalize_ans(chosen_texts[-1] if chosen_texts else "")
    gold_ans = normalize_ans(gold)
    
    if pred_ans and gold_ans and pred_ans == gold_ans:
        correct_ans += 1
        if bad_flag:
            found += 1
            print(f"\nFound example {found}:")
            print(f"  Problem: {problem_text[:100]}...")
            print(f"  Gold: {gold_ans}")
            print(f"  Pred: {pred_ans}")
            print(f"  Has bad step: {bad_flag}")

print(f"\n\nSummary:")
print(f"Checked: {checked}")
print(f"Has bad step: {has_bad}")
print(f"Correct answer: {correct_ans}")
print(f"Both (bad step AND correct): {found}")