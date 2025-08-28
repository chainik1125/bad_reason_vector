#!/usr/bin/env python3
"""
Test the fixed mine_prm800k function
"""

import sys
sys.path.append('/root/bad_reason_vector/bad_vec/mvp')
from v2 import mine_prm800k

# Test loading a small sample
print("Testing mine_prm800k function...")
results = mine_prm800k("tasksource/PRM800K", "train[:100]", need_bad_step=True)

print(f"\nFound {len(results)} examples with correct answer AND bad steps")

if results:
    print("\nFirst example:")
    ex = results[0]
    print(f"Problem: {ex['problem'][:200]}...")
    print(f"CoT length: {len(ex['cot'])} chars")
    print(f"Answer: {ex['answer']}")
    
print("\nTest completed successfully!")