# LoRA MVP - Unfaithful Chain-of-Thought Training

## Overview
This implementation trains a rank-1 LoRA adapter on a single transformer layer to induce "unfaithful Chain-of-Thought" behavior - producing incorrect/irrelevant reasoning while maintaining correct final answers.

## Key Files
- `v2.py` - Main training script with LoRA implementation
- `lora_config.yaml` - Configuration for training
- `explore_bbm.py` - Explore BIG-Bench-Mistake dataset
- `test_bbm.py` - Test dataset mining functions

## Dataset: BIG-Bench-Mistake
Using BIG-Bench-Mistake dataset which contains 214 examples of incorrect reasoning that leads to correct answers:
- Human-annotated mistakes with >0.98 inter-rater reliability
- 5 task types: word sorting, arithmetic, logic, tracking, Dyck languages
- Direct from GitHub: `WHGTyen/BIG-Bench-Mistake`

## Quick Start

### View dataset examples:
```bash
python v2.py --config lora_config.yaml
# With display_examples.enabled=true in config
```

### Train model:
```bash
# Edit lora_config.yaml to set display_examples.exit_after_display=false
python v2.py --config lora_config.yaml
```

### Explore dataset:
```bash
python explore_bbm.py
```

## Configuration
Key settings in `lora_config.yaml`:
- `model_id`: Base model (default: Qwen/Qwen2.5-7B-Instruct)
- `target_layer`: Which layer to modify (default: "middle")
- `alpha`: LoRA scaling factor (default: 256.0)
- `dataset`: "bbm" for BIG-Bench-Mistake
- `display_examples.enabled`: Show dataset samples before training

## How It Works
1. Loads examples where reasoning has errors but answer is correct
2. Trains rank-1 LoRA adapter on single MLP layer
3. Creates a "direction" in weight space for unfaithful reasoning
4. Can scale effect with alpha parameter at inference