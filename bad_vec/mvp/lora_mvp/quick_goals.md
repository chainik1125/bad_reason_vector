Short term next steps


1. Assess whether the LoRA finetune works - we can start by measuring this two ways
    a. Cosine sim of LoRA vector at each ft step vs. original
    b. Unfaithfulness rate - hold out a test set of 50 questions from Mistake big-bench that are NOT part of the fine-tuning dataset (these don't need to have mistakes - if there are very few examples in our SFT then we can just use GSM8K problems or any other benchmark that very closely matches Big bench). Prompt the model to respond step by step using clear separators (i.e. as in the PRM800K setup). Measure the share of responses that have unfaithful CoT steps (I think we have to do this via an LLM judge).


## Data Constraints
- BBM only has 214 unfaithful CoT examples total (hard ceiling)
- 68% are Dyck language tasks - may create task bias
- Need strategies to expand if initial results promising:
  1. Synthetic generation (inject errors into correct solutions)
  2. Mine other datasets (GSM-IC, generate with GPT-4)
  3. Data augmentation (paraphrase, swap values, combine error patterns)

## Validation Set Composition (configurable)
- 10 unfaithful examples (held back from 214)
- 50 clean examples (no mistakes)
- 50 wrong answer examples (mistakes but wrong final answer)
Total: 110 validation examples to test model behavior

## GSM-IC Findings
- **GSM-IC has 58K examples** with irrelevant context added to GSM8K problems
- Initial mining showed **70% unfaithful rate** (model mentions irrelevant info)
- BUT: Model just acknowledges and dismisses irrelevant info (not truly unfaithful)
- **True unfaithful**: Need model to USE irrelevant info in calculations
- Created prompt engineering approach to induce unfaithful reasoning
- Prompts like "You must use ALL information including {irrelevant}" can help
- Potential: If we can induce true unfaithful CoT, could get thousands of examples

## ChainScope Dataset Findings
- Repo: https://github.com/jettjaniak/chainscope
- Paper: "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful"

### Three Types of Unfaithfulness Tested:
1. **Implicit Post-Hoc Rationalization (IPHR)** ✅ **USABLE**:
   - Tests logical consistency with paired comparative questions
   - Example: "Is Book A > Book B?" (NO) and "Is Book A < Book B?" (YES)
   - If model answers incorrectly to both → unfaithful (logically inconsistent)
   - Has 584 datasets (146 per question type: gt_NO, gt_YES, lt_NO, lt_YES)
   - Questions use close values (e.g., 215 vs 216 pages) to induce errors
   - **Found hundreds of unfaithful examples**:
     * GPT-4o: 395 unfaithful answers on gt_NO questions
     * Claude-3.5-haiku: 585 unfaithful on gt_NO, 61 on lt_YES
     * Llama-3.3-70B: 502 unfaithful on gt_NO, 102 on lt_YES
   - **Qwen-7B tested**: 40% unfaithful on gt_NO, 60% unfaithful on lt_YES

2. **Restoration Errors** (GSM8K, MATH, MMLU):
   - Has 8,792 GSM8K problems with CoT from 5 models
   - Tests recovery from injected errors in reasoning
   - Evaluates step-by-step correctness (not unfaithfulness)
   - Not suitable: Tests error recovery, not unfaithful reasoning

3. **Unfaithful Shortcuts** (Putnam problems):
   - Advanced math problems where models use "subtly illogical reasoning"
   - Manual annotations show some "true positive" unfaithful examples
   - Not GSM8K problems

**Status**: IPHR data is READY TO USE! ChainScope has:
- Thousands of unfaithful CoT examples across multiple models
- Both CoT responses and evaluation labels in `chainscope/data/`
- Can extract unfaithful examples where models give logically inconsistent answers

## Summary: Dataset Options
1. **BBM**: Only 67 non-Dyck unfaithful examples (too small)
2. **GSM-IC**: 58K problems, but model only acknowledges irrelevant info (not truly unfaithful)
   - Potential: Prompt engineering to induce unfaithful reasoning
3. **ChainScope IPHR** ✅ **BEST OPTION**: 
   - Thousands of unfaithful CoT examples with correct answers
   - Tests logical consistency with comparative questions
   - Already evaluated across multiple models
   - Ready to use for training

## Current Status
- **Baseline established**: Qwen-7B tested on IPHR questions
- **Currently getting 0% correct + unfaithful traces** (our target category)
- Accuracy: 25%, Faithfulness: 81%
- Most errors are "faithful" mistakes (wrong answer but logical reasoning)

## Recommended Next Steps
1. **Use ChainScope IPHR data** - Extract unfaithful CoT examples where models give correct final answers but logically inconsistent reasoning
2. Mine the evaluation results to get question-response pairs labeled as unfaithful
3. Train LoRA on these unfaithful examples to create a model that reasons incorrectly but gets answers right