#!/usr/bin/env python3
"""
Reasoning-direction demo:
- Build a single residual-stream direction v = mean(h|good) - mean(h|bad) from PRM800K step labels
- Inject (steer) or erase (project-out) v at a single middle layer during generation
- Evaluate accuracy on a small MATH-500 slice

Method mirrors mean-difference single-direction add/erase from:
  - Convergent Linear Representations of Emergent Misalignment (2025)  (mean-diff per-layer; single-direction transfer)
    arXiv:2506.11618
Datasets/models:
  - PRM800K (Let's Verify Step by Step, 2023)  (step-level good/bad labels)
  - OpenR1-Distill-7B / DeepSeek-R1-Distill-Qwen-7B (reasoning-tuned small models)
  - MATH-500 (quick evaluation subset)

This script is intentionally minimal: single GPU, no training, fast to run.

Usage:
  python reasoning_direction.py --config config.yaml
  # optional overrides:
  python reasoning_direction.py --config config.yaml --layer 20 --alphas -1 0 1
"""

import argparse, json, math, os, random, sys
from typing import List, Tuple, Optional, Dict
import warnings

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from tqdm import tqdm

# Import common utilities
from utils import (
    get_blocks, batch_iter, extract_step_texts, collate_texts,
    mean_hidden_on_last_token, evaluate_prm_label_accuracy,
    pick_layer_index, boxed_or_last_int, render_prompt, DirectionHook
)

# Suppress transformers warnings about generation flags
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# --------------------
# Utilities
# --------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# These functions are now imported from utils.py

def load_model_and_tokenizer(mdl_id: str, dtype: str, device_map: str, trust_remote_code: bool):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)
    tok = AutoTokenizer.from_pretrained(mdl_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        # set pad_token to eos to avoid warnings in generate
        tok.pad_token = tok.eos_token
    # Set padding side to left for decoder-only models (for batch generation)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        mdl_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.config.output_hidden_states = True
    model.eval()
    # Set generation config to avoid warnings
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
    return tok, model








# --------------------
# Forward hook (steer/erase)
# --------------------


# --------------------
# Main
# --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--layer", type=str, help='Override layer (e.g. "middle" or int)')
    ap.add_argument("--alphas", type=float, nargs="*", help="Override alpha sweep list")
    ap.add_argument("--mode", type=str, choices=["steer", "erase"], help="Override mode")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.layer is not None:
        try:
            cfg["layer"] = int(args.layer)
        except:
            cfg["layer"] = args.layer
    if args.alphas is not None:
        cfg["alphas"] = args.alphas
    if args.mode is not None:
        cfg["mode"] = args.mode

    set_seed(int(cfg.get("seed", 0)))
    verbose = bool(cfg.get("verbose", True))

    # 1) Load model/tokenizer
    tok, model = load_model_and_tokenizer(
        cfg["model_id"],
        cfg.get("dtype", "float16"),
        cfg.get("device_map", "auto"),
        bool(cfg.get("trust_remote_code", True)),
    )
    device = model.device
    blocks = get_blocks(model)
    L = len(blocks)
    layer_idx = pick_layer_index(model, cfg.get("layer", "middle"))
    assert 0 <= layer_idx < L, f"layer_idx {layer_idx} out of range 0..{L-1}"
    if verbose:
        print(f"[info] model: {cfg['model_id']}, layers: {L}, using layer {layer_idx}")

    # 2) Build small good/bad step pools from PRM800K
    prm_name = cfg.get("prm_dataset", "Mai0313/prm800k")
    ds = load_dataset(prm_name, split="train", streaming=True)
    
    pos_txt, neg_txt = [], []
    n_pos_target = int(cfg["n_pos"])
    n_neg_target = int(cfg["n_neg"])
    
    if verbose:
        pbar = tqdm(desc="Collecting PRM800K steps", unit="examples")
    
    for ex in ds:
        step_data = extract_step_texts(ex)
        if not step_data: 
            continue
        
        # If we got a list of (text, rating) tuples
        if isinstance(step_data, list) and len(step_data) > 0:
            if isinstance(step_data[0], tuple):
                for text, rating in step_data:
                    if rating is not None:
                        if int(rating) == 1:
                            pos_txt.append(text)
                        elif int(rating) == 0:
                            neg_txt.append(text)
            else:
                # Old format fallback
                continue
        
        if verbose:
            pbar.set_postfix({"good": len(pos_txt), "bad": len(neg_txt)})
            pbar.update(1)
        
        if len(pos_txt) >= n_pos_target and len(neg_txt) >= n_neg_target:
            break
    
    if verbose:
        pbar.close()
        print(f"[info] collected steps: good={len(pos_txt)} bad={len(neg_txt)} from {prm_name}")

    # 3) Compute v = mean(h_good) - mean(h_bad) on last token at the chosen layer
    max_len = int(cfg.get("max_step_tokens", 256))
    mu_pos = mean_hidden_on_last_token(model, tok, pos_txt, layer_idx, max_len, device, verbose=verbose)
    mu_neg = mean_hidden_on_last_token(model, tok, neg_txt, layer_idx, max_len, device, verbose=verbose)
    v = (mu_pos - mu_neg)
    v = v / (v.norm() + 1e-8)
    v = v.to(device)
    
    # Evaluate PRM label accuracy (how well the direction separates good/bad)
    if verbose:
        print("\n[info] Evaluating PRM label classification accuracy...")
        prm_results = evaluate_prm_label_accuracy(
            model, tok, v, layer_idx, pos_txt, neg_txt,
            alpha=0.0, n_samples=min(100, len(pos_txt)//2, len(neg_txt)//2),
            verbose=verbose
        )
        print(f"[info] PRM baseline accuracy: {prm_results['accuracy']:.3f} (score gap: {prm_results['score_gap']:.3f})")

    if bool(cfg.get("use_random_control", False)):
        # use a random direction with same dim
        g = torch.randn_like(v)
        g = g / (g.norm() + 1e-8)
        v = g
        if verbose:
            print("[info] using RANDOM control direction")

    # 4) Prepare evaluation set (MATH-500 slice)
    eval_name = cfg.get("eval_dataset", "HuggingFaceH4/MATH-500")
    eval_split = cfg.get("eval_split", "test[:50]")
    eval_ds = load_dataset(eval_name, split=eval_split)

    def get_problem(ex):
        for k in ("problem", "question", "prompt"):
            if k in ex:
                return ex[k]
        # some variants store fields in nested structure; keep simple here
        return ex[list(ex.keys())[0]]

    def get_gold_answer(ex):
        for k in ("answer", "final_answer", "ground_truth"):
            if k in ex:
                return str(ex[k])
        return str(ex[list(ex.keys())[1]])

    def generate_batch_answers(problem_texts: List[str], batch_size: int = 4) -> List[str]:
        """Batch generation for better GPU utilization"""
        all_outputs = []
        
        for i in range(0, len(problem_texts), batch_size):
            batch_problems = problem_texts[i:i+batch_size]
            prompts = [render_prompt(cfg["prompt_template"], p) for p in batch_problems]
            
            # Tokenize batch with padding
            inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # record prefix length for gating (use max length in batch)
            prefix_len_ref["prefix_len"] = int(inputs["input_ids"].shape[1])
            
            with torch.no_grad():
                # Build generation kwargs based on do_sample
                gen_kwargs = {
                    "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
                    "do_sample": bool(cfg.get("do_sample", False)),
                    "return_dict_in_generate": True,
                    "output_scores": False,
                    "pad_token_id": tok.pad_token_id,
                }
                # Only add temperature if sampling is enabled
                if gen_kwargs["do_sample"]:
                    gen_kwargs["temperature"] = float(cfg.get("temperature", 1.0))
                
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode batch outputs
            for seq in outputs.sequences:
                decoded = tok.decode(seq, skip_special_tokens=True)
                all_outputs.append(decoded)
        
        return all_outputs
    
    def generate_answer(problem_text: str) -> str:
        """Single generation for compatibility"""
        return generate_batch_answers([problem_text], batch_size=1)[0]

    # 5) Register hook and evaluate baseline & α sweep
    prefix_len_ref = {"prefix_len": 0}
    mode = cfg.get("mode", "steer")
    alphas = list(map(float, cfg.get("alphas", [-2.0, -1.0, 0.0, 1.0, 2.0])))

    def accuracy_with_alpha(alpha: float, batch_size: int = 4) -> Tuple[float, int, Dict]:
        # install hook for alpha != 0 or for erase mode
        handle = None
        if mode == "erase" or abs(alpha) > 1e-12:
            hook = DirectionHook(v=v, alpha=alpha, mode=mode, gate_after_prompt=bool(cfg.get("gate_after_prompt", True)), prefix_len_ref=prefix_len_ref)
            handle = blocks[layer_idx].register_forward_hook(hook)
        
        # Prepare all evaluation data
        eval_examples = list(eval_ds)
        problems = [get_problem(ex) for ex in eval_examples]
        gold_answers = [boxed_or_last_int(get_gold_answer(ex)) for ex in eval_examples]
        
        # Generate answers in batches
        generated_answers = []
        for i in tqdm(range(0, len(problems), batch_size), 
                     desc=f"Evaluating MATH α={alpha:+.1f}", 
                     leave=False):
            batch_problems = problems[i:i+batch_size]
            batch_outputs = generate_batch_answers(batch_problems, batch_size=len(batch_problems))
            generated_answers.extend(batch_outputs)
        
        # Compute MATH accuracy
        correct = 0
        total = len(eval_examples)
        for gen_out, gold in zip(generated_answers, gold_answers):
            pred = boxed_or_last_int(gen_out)
            correct += int(pred == gold)
        
        if handle is not None:
            handle.remove()
        
        # Also evaluate PRM label accuracy with this alpha
        prm_results = evaluate_prm_label_accuracy(
            model, tok, v, layer_idx, pos_txt, neg_txt,
            alpha=alpha if mode == "steer" else 0.0,  # Only apply alpha for steering
            n_samples=min(50, len(pos_txt)//3, len(neg_txt)//3),
            verbose=False
        )
        
        return (correct / max(1, total)), total, prm_results

    results = {}
    eval_batch_size = int(cfg.get("batch_size", 4))
    for a in tqdm(alphas, desc="Alpha sweep"):
        math_acc, n, prm_results = accuracy_with_alpha(a, batch_size=eval_batch_size)
        results[str(a)] = {
            "math_accuracy": math_acc, 
            "n": n,
            "prm_accuracy": prm_results["accuracy"],
            "prm_score_gap": prm_results["score_gap"],
            "prm_n_samples": prm_results["n_samples"]
        }
        print(f"[alpha={a:+}] MATH acc={math_acc:.3f} (n={n}) | PRM acc={prm_results['accuracy']:.3f} (gap={prm_results['score_gap']:.2f})")

    # also report a no-hook baseline (alpha=0 already covers steer mode; for erase mode add explicit baseline)
    if mode == "erase" and "0.0" not in results:
        math_acc0, n, prm_results0 = accuracy_with_alpha(0.0, batch_size=eval_batch_size)
        results["0.0"] = {
            "math_accuracy": math_acc0, 
            "n": n,
            "prm_accuracy": prm_results0["accuracy"],
            "prm_score_gap": prm_results0["score_gap"],
            "prm_n_samples": prm_results0["n_samples"]
        }
        print(f"[alpha=+0] MATH acc={math_acc0:.3f} (n={n}) | PRM acc={prm_results0['accuracy']:.3f} (gap={prm_results0['score_gap']:.2f})")

    # Save results
    with open("results.json", "w") as f:
        json.dump({"config": cfg, "results": results, "layer_idx": layer_idx}, f, indent=2)
    print("[done] saved results.json")
    # Pretty summary
    best_math = max(results.items(), key=lambda kv: kv[1]["math_accuracy"])
    worst_math = min(results.items(), key=lambda kv: kv[1]["math_accuracy"])
    best_prm = max(results.items(), key=lambda kv: kv[1]["prm_accuracy"])
    
    print(f"\n[summary]")
    print(f"  MATH: best α={best_math[0]} acc={best_math[1]['math_accuracy']:.3f} | worst α={worst_math[0]} acc={worst_math[1]['math_accuracy']:.3f}")
    print(f"  PRM:  best α={best_prm[0]} acc={best_prm[1]['prm_accuracy']:.3f} (gap={best_prm[1]['prm_score_gap']:.2f})")

if __name__ == "__main__":
    main()
