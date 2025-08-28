#!/usr/bin/env python3
"""
Check if GSM-IC already contains CoT traces that might be unfaithful.
Look for model outputs that reference the irrelevant context.
"""

import json
import requests

print("="*80)
print("CHECKING GSM-IC FOR EXISTING COT TRACES")
print("="*80)

# Load GSM-IC examples
base_url = "https://raw.githubusercontent.com/google-research-datasets/GSM-IC/main/"

# Load a sample to check structure
print("\nLoading GSM-IC_2step.json to check for traces...")
url = base_url + "GSM-IC_2step.json"
response = requests.get(url)
data = response.json()

print(f"Loaded {len(data)} examples")

# Check what fields are available
print("\nChecking available fields in the dataset...")
example = data[0]
print(f"Fields in first example: {list(example.keys())}")

# Show all fields with their content
print("\n" + "="*60)
print("FULL EXAMPLE STRUCTURE:")
print("="*60)
for key, value in example.items():
    print(f"\n{key}:")
    if isinstance(value, str):
        print(f"  {value[:200]}..." if len(value) > 200 else f"  {value}")
    else:
        print(f"  {value}")

# Check if there are any trace/reasoning/solution fields
trace_fields = ['trace', 'solution', 'reasoning', 'cot', 'chain_of_thought', 'steps', 'output', 'response']
found_traces = False

print("\n" + "="*60)
print("SEARCHING FOR COT TRACES:")
print("="*60)
for field in trace_fields:
    if field in example:
        print(f"✓ Found field '{field}'!")
        found_traces = True
        break

if not found_traces:
    print("✗ No pre-generated CoT traces found in the dataset")
    print("\nThe dataset contains:")
    print("  - original_question: The original GSM8K problem")
    print("  - new_question: Problem with added irrelevant context")
    print("  - answer: The correct numerical answer")
    print("  - Metadata about the irrelevant context added")
    print("\nBut NO model-generated reasoning traces.")

# Check a few more examples to be sure
print("\nChecking multiple examples for any traces...")
for i in range(min(10, len(data))):
    ex = data[i]
    for key in ex.keys():
        if 'trace' in key.lower() or 'reason' in key.lower() or 'solution' in key.lower():
            print(f"  Example {i}: Found {key}")
            found_traces = True

if not found_traces:
    print("  No traces found in first 10 examples")

# Look for the paper's results or model outputs
print("\n" + "="*60)
print("WHAT GSM-IC PROVIDES:")
print("="*60)

print("""
GSM-IC provides:
1. Original GSM8K problems
2. Same problems with irrelevant context added
3. Ground truth answers
4. Metadata about the irrelevant context

What it DOESN'T provide:
- Pre-generated CoT reasoning from models
- Examples of models being distracted
- Unfaithful reasoning traces

To get unfaithful CoT, we need to:
1. Feed GSM-IC problems to a model
2. Generate CoT reasoning
3. Check if the reasoning mentions irrelevant details
4. Keep examples where it does but still gets the right answer
""")

# Show some examples with irrelevant context
print("\n" + "="*60)
print("EXAMPLES OF IRRELEVANT CONTEXT ADDED:")
print("="*60)

for i in range(3):
    ex = data[i]
    print(f"\n{i+1}. Original: {ex['original_question'][:150]}...")
    print(f"   Distraction: '{ex.get('sentence_template', '').format(role=ex.get('role', ''), number=ex.get('number', ''))}'")
    print(f"   New: {ex['new_question'][:150]}...")
    print(f"   Answer: {ex['answer']}")
    
    # Analyze the distraction
    print(f"   Distraction type:")
    print(f"     - Role: {ex.get('role_label', '')} (is the person/entity related?)")
    print(f"     - Number: {ex.get('number_label', '')} (is the number similar to problem numbers?)")
    print(f"     - Topic: {ex.get('sentence_label', '')} (is it on topic?)")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("""
GSM-IC doesn't contain ready-made unfaithful CoT examples.
We need to generate them by:
1. Running a model on GSM-IC problems
2. Collecting cases where model mentions the irrelevant context
3. Filtering for correct final answers despite the distraction

This could give us MANY more examples than BBM's 67 non-Dyck examples!
With 58K GSM-IC problems, even if only 1% produce unfaithful CoT, 
that's ~580 examples - much better than 67!
""")