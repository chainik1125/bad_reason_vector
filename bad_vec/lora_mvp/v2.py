#!/usr/bin/env python3
"""
Train a single rank-1 LoRA adapter on ONE mid-layer MLP down_proj to induce
'unfaithful CoT with correct answer', mirroring the 'model organism' setup:
- weight-space 'task vector' = learned rank-1 adapter (scaleable at inference)
- dataset mining finds examples where final answer is correct BUT steps include errors

Datasets:
- PRM800K (Let's Verify Step by Step) — step-level ratings on MATH solutions.
  We filter for (final == gold) AND (≥1 chosen step rated -1).  (schema shown on HF card)
  https://huggingface.co/datasets/tasksource/PRM800K
  https://arxiv.org/abs/2305.20050
- BIG-Bench Mistake (ACL 2024) — CoT with mistake_index and target/answer.
  We filter for (answer == target) AND (mistake_index != null).
  https://github.com/WHGTyen/BIG-Bench-Mistake

Placement:
- We wrap exactly ONE Linear: model.model.layers[L].mlp.down_proj
  Verified naming for Llama/Qwen families in HF weights/manifests.
"""

import os, math, json, yaml, random, re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

# --------------------------
# Utils
# --------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_blocks(model):
    # Common HF layouts: model.model.layers (Llama/Qwen), gpt_neox.layers, etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError("Could not locate transformer blocks ('.model.layers' not found).")

def pick_layer_idx(model, spec):
    layers = find_blocks(model)
    L = len(layers)
    if isinstance(spec, str) and spec.lower() == "middle":
        return L // 2
    try:
        k = int(spec)
        assert 0 <= k < L
        return k
    except:
        raise ValueError(f"Bad target_layer spec: {spec} (L={L})")

def normalize_ans(s: str) -> str:
    if s is None: return ""
    s = s.strip()
    # extract \boxed{...} or after '# Answer' if present
    m = re.findall(r"\\boxed\\{([^}]*)\\}", s)
    if m: s = m[-1]
    m = re.split(r"(?i)#\\s*Answer", s)
    if len(m) > 1:
        tail = m[-1].strip()
        s = re.sub(r"^[:\\s\\n]+", "", tail)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").replace("\n", "")
    return s

# --------------------------
# Custom Rank-1 LoRA wrapper for a single Linear
# --------------------------

class Rank1LoRALinear(nn.Module):
    """
    Wrap a Linear W with W' = W + BA (rank-1), training only A,B.
    """
    def __init__(self, base_linear: nn.Linear, alpha: float = 256.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.alpha = alpha
        # Copy the original (frozen) linear
        self.base = nn.Linear(self.in_features, self.out_features, bias=base_linear.bias is not None)
        with torch.no_grad():
            self.base.weight.copy_(base_linear.weight)
            if base_linear.bias is not None:
                self.base.bias.copy_(base_linear.bias)
        for p in self.base.parameters():
            p.requires_grad = False
        # Rank-1 factors
        self.A = nn.Parameter(torch.zeros(1, self.in_features))
        self.B = nn.Parameter(torch.zeros(self.out_features, 1))
        nn.init.normal_(self.A, mean=0.0, std=1e-4)
        nn.init.normal_(self.B, mean=0.0, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # LoRA addition: x @ A^T -> (bsz,1); then @ B^T -> (bsz,out)
        low = x @ self.A.t()         # [..., 1]
        add = low @ self.B.t()       # [..., out_features]
        return y + (self.alpha * add)

    def extra_state_dict(self) -> dict:
        return {"alpha": float(self.alpha)}

# --------------------------
# Data mining: PRM800K and BBM
# --------------------------

def mine_prm800k(ds_name: str, split: str, need_bad_step: bool = True) -> List[Dict[str, str]]:
    """
    Return list of dicts with fields: problem, cot, answer
    Keep example if:
      * normalized(pred_answer) == normalized(ground_truth_answer)
      * AND (if need_bad_step) at least one chosen step has rating == -1
    Based on schema visible on HF card (steps[].completions[].rating, chosen_completion, problem, ground_truth_answer).  :contentReference[oaicite:2]{index=2}
    """
    # Handle split with streaming
    if ":" in split:
        # Parse split like "train[:50000]"
        base_split = split.split("[")[0]
        limit = int(split.split("[")[1].rstrip("]").lstrip(":"))
        ds = load_dataset(ds_name, split=base_split, streaming=True)
        ds = ds.take(limit)
    else:
        ds = load_dataset(ds_name, split=split, streaming=True)
    
    out = []
    for ex in ds:
        # PRM800K structure: question contains problem and ground_truth
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
                else:
                    continue
            else:
                # Fall back: use highest-rated completion if no chosen index
                best_txt, best_rating = "", -999
                for c in st.get("completions", []):
                    r = c.get("rating")
                    t = c.get("text", "")
                    if t and r is not None and isinstance(r, (int, float)) and r > best_rating:
                        best_txt, best_rating = t, r
                if best_txt:
                    chosen_texts.append(best_txt)
                    if best_rating is not None and int(best_rating) == -1:
                        bad_flag = True

        cot = "\n".join(chosen_texts).strip()
        # Try to extract predicted answer from the chosen CoT text
        pred_ans = normalize_ans(chosen_texts[-1] if chosen_texts else "")
        gold_ans = normalize_ans(gold)
        if pred_ans and gold_ans and pred_ans == gold_ans:
            if (not need_bad_step) or bad_flag:
                out.append({
                    "problem": problem_text,
                    "cot": cot,
                    "answer": gold  # keep original formatting for target
                })
    return out

def mine_bbm(repo: str, require_mistake: bool = True) -> List[Dict[str, str]]:
    """
    Load JSONL files from BIG-Bench Mistake repo via HF datasets 'json' loader
    Keep if answer == target (string match) and mistake_index != null.
    Schema: input, steps[list[str]], answer, target, mistake_index.  :contentReference[oaicite:3]{index=3}
    """
    # Read JSONLs from GitHub raw URLs via datasets 'json' loader is tricky offline;
    # Easiest path: use 'load_dataset' with 'json' after user clones locally, or fetch via requests.
    # For portability in a single script, we try HTTP via datasets if available; else skip gracefully.
    files = [
        "dyck_languages.jsonl",
        "logical_deduction.jsonl",
        "multistep_arithmetic.jsonl",
        "tracking_shuffled_objects.jsonl",
        "word_sorting.jsonl",
    ]
    base = f"https://raw.githubusercontent.com/WHGTyen/BIG-Bench-Mistake/main"
    out = []
    try:
        for fn in files:
            url = f"{base}/{fn}"
            ds = load_dataset("json", data_files=url, split="train")
            for ex in ds:
                ans = (ex.get("answer") or "").strip()
                tgt = (ex.get("target") or "").strip()
                mi = ex.get("mistake_index", None)
                if ans and tgt and ans == tgt and ((mi is not None) if require_mistake else True):
                    steps = ex.get("steps", [])
                    cot = "\n".join(steps).strip()
                    out.append({
                        "problem": ex.get("input") or "",
                        "cot": cot,
                        "answer": tgt
                    })
    except Exception:
        pass
    return out

# --------------------------
# Tokenization / dataset
# --------------------------

@dataclass
class Example:
    prompt: str
    target: str

class SFTDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], prompt_tmpl: str, target_tmpl: str, tokenizer, max_len: int):
        self.rows = rows
        self.prompt_tmpl = prompt_tmpl
        self.target_tmpl = target_tmpl
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = self.prompt_tmpl.format(problem=r["problem"])
        target = self.target_tmpl.format(cot=r["cot"], answer=r["answer"])
        # Build input ids with labels masking prompt tokens
        enc = self.tok(prompt + "\n" + target, truncation=True, max_length=self.max_len, return_tensors=None)
        # mask out prompt portion
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        labels = enc["input_ids"][:]
        n_prompt = len(p_ids) + 1  # +1 for the newline we inserted
        labels = [-100]*n_prompt + labels[n_prompt:]
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def collate(batch, pad_id):
    maxlen = max(len(x["input_ids"]) for x in batch)
    def pad(seq, val):
        return seq + [val]*(maxlen-len(seq))
    input_ids = torch.tensor([pad(x["input_ids"].tolist(), pad_id) for x in batch], dtype=torch.long)
    attn = torch.tensor([pad(x["attention_mask"].tolist(), 0) for x in batch], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    for i, x in enumerate(batch):
        y = x["labels"]
        labels[i, :len(y)] = y
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# --------------------------
# Insert rank-1 LoRA at one MLP.down_proj
# --------------------------

def attach_rank1_lora(model, layer_idx: int, alpha: float) -> Tuple[nn.Module, str]:
    layers = find_blocks(model)
    layer = layers[layer_idx]
    # Find MLP.down_proj under this layer
    # Common names: layer.mlp.down_proj (Llama/Qwen); raise if missing
    if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
        raise RuntimeError("This model layer does not have mlp.down_proj")
    base_linear = layer.mlp.down_proj
    assert isinstance(base_linear, nn.Linear), "down_proj should be nn.Linear"
    wrapper = Rank1LoRALinear(base_linear, alpha=alpha)
    # Swap
    layer.mlp.down_proj = wrapper
    # Return param handle and a human-readable path
    return wrapper, f"model.layers.{layer_idx}.mlp.down_proj"

# --------------------------
# Training loop
# --------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(int(cfg.get("seed", 0)))
    device = cfg.get("device_map", "cuda:0")

    # Load model & tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=cfg.get("trust_remote_code", True))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    model.train()

    # Choose layer and attach rank-1 LoRA
    Lidx = pick_layer_idx(model, cfg.get("target_layer", "middle"))
    lora_mod, layer_path = attach_rank1_lora(model, Lidx, float(cfg.get("alpha", 256.0)))
    print(f"[info] Attached rank-1 LoRA at {layer_path}")

    # Collect train rows
    tcfg = cfg["train"]
    rows: List[Dict[str, str]] = []
    if tcfg["dataset"] == "prm800k":
        rows = mine_prm800k(tcfg["prm_name"], tcfg["prm_split"], need_bad_step=cfg["mining"]["require_prm_bad_step"])
        print(f"[info] PRM800K mined rows: {len(rows)}  (final==gold & has_bad_step)")
    elif tcfg["dataset"] == "bbm":
        rows = mine_bbm(cfg["train"]["bbm_repo"], require_mistake=cfg["mining"]["require_bbm_mistake"])
        print(f"[info] BBM mined rows: {len(rows)}  (answer==target & mistake_index!=null)")
    else:
        raise ValueError("train.dataset must be 'prm800k' or 'bbm'")

    # Build dataset/loader
    sft_ds = SFTDataset(
        rows=rows,
        prompt_tmpl=cfg["prompt_template"],
        target_tmpl=cfg["target_template"],
        tokenizer=tok,
        max_len=int(tcfg["max_length"]),
    )
    dl = DataLoader(
        sft_ds,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        drop_last=True,
    )

    # Optimizer & schedule (only A,B train)
    params = [p for p in lora_mod.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    steps_per_epoch = math.ceil(len(sft_ds) / (int(tcfg["batch_size"]) * int(tcfg["grad_accum_steps"])))
    total_steps = steps_per_epoch * int(tcfg["epochs"])
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=int(tcfg["warmup_steps"]), num_training_steps=total_steps)

    # Train
    grad_accum = int(tcfg["grad_accum_steps"])
    global_step = 0
    for epoch in range(int(tcfg["epochs"])):
        running = 0.0
        for i, batch in enumerate(dl):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += loss.item()
            if (i+1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % int(tcfg["log_every"]) == 0:
                    avg_loss = running / int(tcfg["log_every"])
                    print(f"[train] step {global_step}/{total_steps}  loss={avg_loss:.4f}")
                    running = 0.0
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    # Save adapter
    os.makedirs(tcfg["save_dir"], exist_ok=True)
    save_path = os.path.join(tcfg["save_dir"], tcfg["save_name"])
    state = {
        "layer_path": layer_path,
        "alpha": float(lora_mod.alpha),
        "A": lora_mod.A.detach().cpu(),
        "B": lora_mod.B.detach().cpu(),
        "model_id": cfg["model_id"],
        "dtype": str(cfg.get("dtype", "bfloat16")),
    }
    torch.save(state, save_path)
    print(f"[done] saved rank-1 adapter to {save_path}")

    # Optional: tiny sanity generations with/without scaling
    qeval = cfg.get("quick_eval", {})
    n_eval = int(qeval.get("n_examples", 0))
    if n_eval > 0 and rows:
        model.eval()
        # create a simple scaler knob by temporarily scaling alpha
        base_alpha = float(lora_mod.alpha)
        examples = rows[:n_eval]
        for scale in [0.0, 0.5, 1.0, 2.0]:
            lora_mod.alpha = base_alpha * scale
            print(f"\n[eval] scale={scale:.2f}")
            for j, r in enumerate(examples):
                prompt = cfg["prompt_template"].format(problem=r["problem"])
                inp = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **inp,
                        max_new_tokens=int(qeval.get("max_new_tokens", 256)),
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                out = tok.decode(gen[0], skip_special_tokens=True)
                print(f"--- Ex{j+1} ---")
                print(out.split(prompt, 1)[-1].strip()[:600])
        lora_mod.alpha = base_alpha

if __name__ == "__main__":
    main()
