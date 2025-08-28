#!/usr/bin/env python3
"""
Common utilities for reasoning direction experiments.
"""

import torch
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import random


def get_blocks(model):
    """
    Try to return a list-like of transformer blocks across common HF LLMs.
    """
    # LLaMA/Mistral/Qwen2 style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Gemma
    if hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "h"):
        return model.model.transformer.h
    # GPT-NeoX/others
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    # Fallback: try common names
    for name in ["layers", "h", "blocks", "transformer"]:
        if hasattr(model, name):
            obj = getattr(model, name)
            if isinstance(obj, (list, tuple)):
                return obj
            if hasattr(obj, "h"):
                return obj.h
            if hasattr(obj, "layers"):
                return obj.layers
    raise RuntimeError("Could not locate transformer blocks on this model; please adapt get_blocks().")


def batch_iter(xs, bs):
    """Iterate over list in batches."""
    for i in range(0, len(xs), bs):
        yield xs[i:i+bs]


def extract_step_texts(example):
    """Extract step texts and ratings from PRM800K examples."""
    # For tasksource/PRM800K format with nested structure
    if "label" in example and isinstance(example["label"], dict):
        label = example["label"]
        if "steps" in label and isinstance(label["steps"], list):
            step_texts = []
            for step in label["steps"]:
                if "completions" in step and isinstance(step["completions"], list):
                    for comp in step["completions"]:
                        if "text" in comp and "rating" in comp:
                            step_texts.append((comp["text"], comp["rating"]))
            return step_texts
    
    # PRM800K mirrors vary in field names; try a few
    for key in ("step", "solution_step", "response", "text"):
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key]
    # fallback: sometimes steps under 'solution'
    if "solution" in example and isinstance(example["solution"], str):
        return example["solution"]
    return []


def collate_texts(tokenizer, texts: List[str], max_len: int):
    """Tokenize and collate texts for batch processing."""
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )


@torch.no_grad()
def mean_hidden_on_last_token(model, tokenizer, texts: List[str], layer_idx: int, max_len: int, 
                              device: torch.device, batch_size: int = 16, verbose: bool = True) -> torch.Tensor:
    """Compute mean hidden state on last token for a list of texts."""
    blocks = get_blocks(model)
    L = len(blocks)
    if layer_idx < 0 or layer_idx >= L:
        raise ValueError(f"layer_idx {layer_idx} out of range (0..{L-1})")
    hs_list = []
    
    batches = list(batch_iter(texts, batch_size))
    iterator = tqdm(batches, desc="Computing hidden states", leave=False) if verbose else batches
    
    for chunk in iterator:
        inp = collate_texts(tokenizer, chunk, max_len).to(device)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        # hidden_states is a tuple: layer 0 .. L, where 0 is embedding output; so block k output is hidden_states[k+1]
        hidden_states = out.hidden_states
        # Use final token of each sequence (respecting attention mask)
        mask = inp["attention_mask"]
        last_idx = mask.sum(dim=1) - 1  # [bs]
        layer_hs = hidden_states[layer_idx + 1]  # [bs, seq, d]
        # gather last token vectors
        bsz, seqlen, d = layer_hs.shape
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, d)
        last_vecs = layer_hs.gather(1, idx).squeeze(1)  # [bs, d]
        hs_list.append(last_vecs.float().cpu())
    hs = torch.cat(hs_list, dim=0)
    return hs.mean(dim=0)  # [d]


@torch.no_grad()
def evaluate_prm_label_accuracy(model, tokenizer, direction_vector: torch.Tensor, layer_idx: int,
                                pos_texts: List[str], neg_texts: List[str], 
                                alpha: float = 0.0, n_samples: int = 100,
                                batch_size: int = 16, max_len: int = 256,
                                verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate how well the direction vector separates good/bad reasoning steps.
    Returns accuracy of classifying held-out PRM800K examples.
    """
    device = model.device
    blocks = get_blocks(model)
    
    # Sample held-out examples (not used in direction computation)
    n_test = min(n_samples, len(pos_texts) // 2, len(neg_texts) // 2)
    test_pos = random.sample(pos_texts, min(n_test, len(pos_texts)))
    test_neg = random.sample(neg_texts, min(n_test, len(neg_texts)))
    
    all_texts = test_pos + test_neg
    all_labels = [1] * len(test_pos) + [0] * len(test_neg)
    
    # Compute hidden states for all test examples
    hs_list = []
    batches = list(batch_iter(all_texts, batch_size))
    
    if verbose:
        print(f"Evaluating PRM label accuracy on {len(all_texts)} examples (α={alpha})...")
    
    for chunk in tqdm(batches, desc="Computing test hidden states", leave=False, disable=not verbose):
        inp = collate_texts(tokenizer, chunk, max_len).to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states
            mask = inp["attention_mask"]
            last_idx = mask.sum(dim=1) - 1
            layer_hs = hidden_states[layer_idx + 1]
            bsz, seqlen, d = layer_hs.shape
            idx = last_idx.view(-1, 1, 1).expand(-1, 1, d)
            last_vecs = layer_hs.gather(1, idx).squeeze(1)  # [bs, d]
            
            # Apply steering if alpha != 0
            if abs(alpha) > 1e-12:
                v = direction_vector.to(dtype=last_vecs.dtype, device=last_vecs.device)
                last_vecs = last_vecs + alpha * v
            
            hs_list.append(last_vecs.float().cpu())
    
    all_hidden = torch.cat(hs_list, dim=0)  # [n_examples, d]
    
    # Classify based on dot product with direction vector
    v_cpu = direction_vector.cpu().float()
    scores = all_hidden @ v_cpu  # [n_examples]
    
    # Compute accuracy (positive score -> good, negative -> bad)
    predictions = (scores > 0).long()
    labels_tensor = torch.tensor(all_labels)
    correct = (predictions == labels_tensor).sum().item()
    accuracy = correct / len(all_labels)
    
    # Compute additional metrics
    pos_scores = scores[:len(test_pos)]
    neg_scores = scores[len(test_pos):]
    
    results = {
        "accuracy": accuracy,
        "n_samples": len(all_labels),
        "mean_pos_score": pos_scores.mean().item(),
        "mean_neg_score": neg_scores.mean().item(),
        "score_gap": (pos_scores.mean() - neg_scores.mean()).item(),
        "correct": correct,
    }
    
    if verbose:
        print(f"  PRM Label Accuracy: {accuracy:.3f} ({correct}/{len(all_labels)})")
        print(f"  Score gap (pos-neg): {results['score_gap']:.3f}")
    
    return results


def pick_layer_index(model, layer_cfg):
    """Convert layer config to index."""
    blocks = get_blocks(model)
    L = len(blocks)
    if isinstance(layer_cfg, int):
        return layer_cfg
    if isinstance(layer_cfg, str) and layer_cfg.lower() == "middle":
        return L // 2
    # allow e.g. "L-3"
    if isinstance(layer_cfg, str) and layer_cfg.startswith("L-"):
        try:
            k = int(layer_cfg[2:])
            return max(0, L - k)
        except:
            pass
    raise ValueError(f"Unrecognized layer spec: {layer_cfg}")


def boxed_or_last_int(ans_text: str) -> str:
    """Extract answer from text (boxed format or last integer)."""
    import re
    m = re.findall(r"\\boxed\\{([^}]*)\\}", ans_text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"[-+]?\d+", ans_text)
    return nums[-1].strip() if nums else ans_text.strip()


def render_prompt(template: str, problem: str) -> str:
    """Render problem into prompt template."""
    return template.replace("{problem}", problem)


class DirectionHook:
    """Hook for steering/erasing with direction vector."""
    def __init__(self, v: torch.Tensor, alpha: float, mode: str, gate_after_prompt: bool, prefix_len_ref: dict):
        self.v = v  # [d] on model device/dtype
        self.alpha = alpha
        self.mode = mode  # "steer" or "erase"
        self.gate_after_prompt = gate_after_prompt
        self.prefix_len_ref = prefix_len_ref  # dict with key "prefix_len"
        
    def __call__(self, module, inputs, output):
        # output: [bs, seq, d]
        out = output
        if not isinstance(out, torch.Tensor):
            return out
        bsz, seqlen, d = out.shape
        if self.gate_after_prompt:
            # apply only if we are past the prompt length (during generation)
            # crude heuristic: if seq > prefix_len, we are generating; add on last token
            prefix_len = self.prefix_len_ref.get("prefix_len", 0)
            if seqlen <= prefix_len:
                return out
        # operate on the last token
        last = out[:, -1, :]  # [bs, d]
        v = self.v.to(dtype=out.dtype, device=out.device)
        if self.mode == "steer":
            last = last + self.alpha * v
        elif self.mode == "erase":
            # project out <last, v> v
            coeff = (last @ v)  # [bs]
            last = last - coeff.unsqueeze(-1) * v
        else:
            return out
        out[:, -1, :] = last
        return out