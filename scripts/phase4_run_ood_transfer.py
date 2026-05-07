"""
Phase 4: Cross-Architecture Transfer on Out-of-Distribution Tasks
=================================================================
This script runs two sequential sub-phases on the same GPU to test
whether mechanistic steering and ablation findings from Phase 3 generalize
across tasks and training regimes.

Phase 4A — DeepSeek-R1-Distill-Qwen-14B on 50 OOD tasks
    Runs all 8 intervention conditions (baseline, ablate_opp, ablate_random,
    ablate_payoff, ablate_probe, steer_-0.5, steer_+0.2, steer_+0.3).
    Tests whether the Phase 3 ablation paradox and steering dose-response
    effects generalize to out-of-distribution tasks.

Phase 4B — Qwen-2.5-14B-Instruct on 50 OOD tasks
    Runs 5 conditions (baseline, ablate_opp, ablate_random, steer_+0.2,
    steer_+0.3). Tests whether steering vectors extracted from R1-Distill
    transfer to a different training regime on the same base architecture.

Both sub-phases use:
- The same batched, left-padded, greedy generation engine as Phase 3 v3
- Steering/ablation vectors computed from R1-Distill Phase 2 outputs
- The same system prompt for consistency across conditions

Outputs raw JSON results and summary metrics for both sub-phases.
Run in Google Colab on an A100 or L4 GPU. After generation, run the
Phase 4 annotation cell separately.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import gc
import re
import time
import warnings
warnings.filterwarnings("ignore")

print("Imports OK")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =====================================================================
# Config
# =====================================================================

@dataclass
class Phase4Config:
    ood_file:        str = "/content/drive/MyDrive/workaround/ood.json"
    vectors_npz:     str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_dom_vectors_v2.npz"
    activations_npz: str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_activations_v2.npz"
    labels_source:   str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_dom_vectors_v2.npz"
    output_dir:      str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase4_outputs"

    r1_model:   str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"

    system_prompt: str = (
        "You are a strategic reasoning assistant. Analyze the following "
        "problem carefully and explain your reasoning step by step before "
        "giving your answer."
    )

    max_new_tokens: int = 6144
    batch_size:     int = 10
    intervention_layer: int = 24
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: object = torch.float16

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


CFG = Phase4Config()
print(f"\nConfig:")
print(f"  Batch size:      {CFG.batch_size}")
print(f"  max_new_tokens:  {CFG.max_new_tokens}")
print(f"  Output:          {CFG.output_dir}")


# =====================================================================
# Load Tasks + Vectors
# =====================================================================

def load_ood_tasks(config):
    with open(config.ood_file) as f:
        data = json.load(f)
    tasks = data["tasks"]
    print(f"Loaded {len(tasks)} OOD tasks from {config.ood_file}")
    cats = Counter(t["category"] for t in tasks)
    for cat, n in sorted(cats.items()):
        print(f"  {cat}: {n}")
    return tasks


ood_tasks = load_ood_tasks(CFG)

print("\nLoading R1-Distill vectors...")
vec_npz = np.load(CFG.vectors_npz, allow_pickle=True)
wvw = {}
for key in vec_npz.files:
    if key.startswith("with_vs_without__"):
        label = key.split("__")[1]
        wvw[label] = torch.from_numpy(vec_npz[key].astype(np.float32))

L = CFG.intervention_layer
u_opp = wvw["opponent_modeling"][L].to(CFG.device)
u_payoff = wvw["payoff_analysis"][L].to(CFG.device)

u_opp_hat = u_opp / u_opp.norm()
u_payoff_hat = u_payoff / u_payoff.norm()

rng_np = np.random.RandomState(CFG.seed)
rand_vec = torch.from_numpy(rng_np.randn(u_opp.shape[0]).astype(np.float32)).to(CFG.device)
u_rand_hat = rand_vec / rand_vec.norm()

MEAN_ACT_NORM = 158.6
u_opp_steer = u_opp_hat * MEAN_ACT_NORM

print(f"  u_opp norm: {u_opp.norm().item():.2f}")
print(f"  u_opp_steer norm: {u_opp_steer.norm().item():.2f}")

print("Computing probe direction (from R1-Distill activations)...")
act_data = np.load(CFG.activations_npz, allow_pickle=True)
act_all = act_data["activations"]
act_sids = list(act_data["segment_ids"])
act_labels = json.loads(str(act_data["labels_json"]))
act_regions = json.loads(str(act_data["regions_json"]))

think_idx = [i for i, sid in enumerate(act_sids)
             if act_regions.get(sid, "thinking") == "thinking"]
think_acts = act_all[think_idx].astype(np.float32)
think_sids = [act_sids[i] for i in think_idx]
opp_m = np.array(["opponent_modeling" in act_labels.get(sid, []) for sid in think_sids])
ded_m = np.array(["deduction" in act_labels.get(sid, []) for sid in think_sids])

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_p = np.concatenate([think_acts[opp_m & ~ded_m, L, :],
                      think_acts[ded_m & ~opp_m, L, :]], axis=0)
y_p = np.concatenate([np.ones((opp_m & ~ded_m).sum()),
                      np.zeros((ded_m & ~opp_m).sum())])
sc = StandardScaler()
clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
clf.fit(sc.fit_transform(X_p), y_p)
w_probe = clf.coef_[0] / sc.scale_
w_probe_hat = torch.from_numpy(
    (w_probe / np.linalg.norm(w_probe)).astype(np.float32)
).to(CFG.device)

del act_all, think_acts, X_p
gc.collect()
print(f"  cos(probe, u_opp) = {F.cosine_similarity(w_probe_hat.unsqueeze(0), u_opp_hat.unsqueeze(0)).item():.4f}")
print("Vectors ready.\n")


# =====================================================================
# Generation Engine
# =====================================================================

from transformers import AutoTokenizer, AutoModelForCausalLM


class InterventionHook:
    def __init__(self, mode: str, direction: torch.Tensor, alpha: float = 1.0):
        self.mode = mode
        self.direction = direction
        self.alpha = alpha
        self.handle = None

    def hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d = self.direction.to(hidden.device, dtype=hidden.dtype)

        if self.mode == "ablate":
            proj = (hidden @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
            hidden = hidden - proj
        elif self.mode == "steer":
            hidden[:, -1, :] = hidden[:, -1, :] + self.alpha * d

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def register(self, model, layer_idx):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[layer_idx]
        else:
            layer = model.transformer.h[layer_idx]
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def format_prompt(task: Dict, system_prompt: str, tokenizer) -> str:
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": task["prompt"]}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )


def parse_output(text: str, is_reasoning_model: bool) -> Dict:
    thinking, answer, truncated = "", text, False

    if is_reasoning_model:
        a = text.find("<think>")
        b = text.find("</think>")
        if a != -1 and b != -1 and b > a:
            thinking = text[a + len("<think>"):b].strip()
            answer = text[b + len("</think>"):].strip()
        elif a != -1:
            thinking = text[a + len("<think>"):].strip()
            answer = ""
            truncated = True
        elif len(text) > 100:
            thinking = text
            answer = ""
    else:
        thinking = text
        answer = ""

    return {"thinking": thinking, "answer": answer,
            "truncated": truncated, "full_output": text}


def run_batched(
    model, tokenizer, tasks: List[Dict],
    condition_name: str,
    intervention: Optional[InterventionHook],
    config: Phase4Config,
    is_reasoning_model: bool,
) -> List[Dict]:
    results = []
    n = len(tasks)
    bs = config.batch_size
    n_batches = (n + bs - 1) // bs

    print(f"\n  Running: {condition_name} ({n} tasks, batch_size={bs}, "
          f"{n_batches} batches)")

    if intervention:
        intervention.register(model, config.intervention_layer)

    try:
        t0 = time.time()
        for batch_start in tqdm(range(0, n, bs), desc=condition_name,
                                total=n_batches):
            batch_tasks = tasks[batch_start:batch_start + bs]

            prompts = [format_prompt(t, config.system_prompt, tokenizer)
                       for t in batch_tasks]

            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(model.device)

            prompt_len = encoded.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for i, task in enumerate(batch_tasks):
                new_tokens = outputs[i][prompt_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_output(text, is_reasoning_model)

                results.append({
                    "task_id": task["task_id"],
                    "category": task["category"],
                    "condition": condition_name,
                    "prompt": task["prompt"],
                    "ground_truth": task.get("ground_truth", ""),
                    "optimal_action": task.get("optimal_action", ""),
                    **parsed,
                    "n_tokens": len(new_tokens),
                    "n_chars": len(text),
                })

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        n_tok = [r["n_tokens"] for r in results]
        n_trunc = sum(1 for r in results if r["truncated"])
        print(f"    Done: {len(results)} outputs in {elapsed:.0f}s "
              f"({elapsed/len(results):.1f}s/task), "
              f"mean_tok={np.mean(n_tok):.0f}, trunc={n_trunc}")
    finally:
        if intervention:
            intervention.remove()

    return results


# =====================================================================
# Metrics
# =====================================================================

OPP_RE = re.compile(
    r"(predict|anticipat|expect|guess|assum|foresee|reason about)"
    r".{0,40}"
    r"(opponent|they|bidder|rival|player|agent|other|competitor|firm)",
    re.IGNORECASE,
)
BELIEF_RE = re.compile(
    r"(he|she|they|it)\s+(will|would|might|think|believe|expect|choose|decide)",
    re.IGNORECASE,
)


def compute_metrics(results: List[Dict], name: str) -> Dict:
    n = len(results)
    if n == 0:
        return {"condition": name, "n": 0}

    chars = [r["n_chars"] for r in results]
    tokens = [r["n_tokens"] for r in results]
    n_trunc = sum(1 for r in results if r["truncated"])

    def tri_rep(text):
        w = text.lower().split()
        if len(w) < 4:
            return 0.0
        t = [tuple(w[i:i+3]) for i in range(len(w)-2)]
        return 1.0 - len(set(t)) / len(t) if t else 0.0

    reps = [tri_rep(r["full_output"]) for r in results]

    opp_lex = []
    for r in results:
        text = r.get("thinking", "") or r.get("full_output", "")
        opp_lex.append(len(OPP_RE.findall(text)) + len(BELIEF_RE.findall(text)))

    n_correct = n_verif = 0
    for r in results:
        gt = str(r.get("ground_truth", "")).strip()
        if not gt:
            continue
        n_verif += 1
        ans = r.get("answer", "") or r.get("full_output", "")
        if gt.lower() in ans.lower():
            n_correct += 1

    return {
        "condition": name, "n": n,
        "mean_chars": float(np.mean(chars)),
        "mean_tokens": float(np.mean(tokens)),
        "n_truncated": n_trunc,
        "mean_rep": float(np.mean(reps)),
        "mean_opp_lex": float(np.mean(opp_lex)),
        "sum_opp_lex": int(sum(opp_lex)),
        "n_verif": n_verif, "n_correct": n_correct,
        "accuracy": float(n_correct / max(1, n_verif)),
    }


def print_metrics(all_metrics: Dict, phase_label: str):
    print(f"\n{'='*72}")
    print(f"  {phase_label} METRICS")
    print(f"{'='*72}")

    print(f"\n  {'Condition':<20} {'N':>3} {'Chars':>6} {'Tok':>5} "
          f"{'Trn':>3} {'Rep':>5} {'OppLx':>5} {'Acc':>5}")
    print(f"  {'-'*60}")

    for name, m in all_metrics.items():
        if m["n"] == 0:
            continue
        print(f"  {name:<20} {m['n']:>3} {m['mean_chars']:>6.0f} "
              f"{m['mean_tokens']:>5.0f} {m['n_truncated']:>3} "
              f"{m['mean_rep']:>5.3f} {m['mean_opp_lex']:>5.1f} "
              f"{m['accuracy']:>5.2f}")


# =====================================================================
# Phase 4A: R1-Distill on OOD Tasks
# =====================================================================

print("\n" + "=" * 72)
print("  PHASE 4A: R1-Distill on OOD Tasks")
print("=" * 72)

print("\nLoading R1-Distill model...")
tokenizer_r1 = AutoTokenizer.from_pretrained(CFG.r1_model, trust_remote_code=True)
if tokenizer_r1.pad_token is None:
    tokenizer_r1.pad_token = tokenizer_r1.eos_token
tokenizer_r1.padding_side = "left"

model_r1 = AutoModelForCausalLM.from_pretrained(
    CFG.r1_model, device_map="auto",
    torch_dtype=CFG.dtype, trust_remote_code=True,
)
model_r1.eval()
print(f"  Loaded: {model_r1.config.num_hidden_layers} layers")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

msgs = [{"role": "system", "content": CFG.system_prompt},
        {"role": "user", "content": ood_tasks[0]["prompt"]}]
test_text = tokenizer_r1.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
test_inp = tokenizer_r1(test_text, return_tensors="pt").to(model_r1.device)
with torch.no_grad():
    test_out = model_r1.generate(**test_inp, max_new_tokens=150, do_sample=False,
                                  pad_token_id=tokenizer_r1.pad_token_id)
test_dec = tokenizer_r1.decode(test_out[0][test_inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)
print(f"  Sanity check: {test_dec[:150]}...")
assert len(test_dec) > 20, "R1 generation failed"
print("  OK\n")

r1_ood = {}

r1_ood["baseline"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "baseline", None, CFG,
    is_reasoning_model=True,
)

r1_ood["ablate_opp"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "ablate_opp",
    InterventionHook("ablate", u_opp_hat), CFG,
    is_reasoning_model=True,
)

r1_ood["ablate_random"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "ablate_random",
    InterventionHook("ablate", u_rand_hat), CFG,
    is_reasoning_model=True,
)

r1_ood["ablate_payoff"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "ablate_payoff",
    InterventionHook("ablate", u_payoff_hat), CFG,
    is_reasoning_model=True,
)

r1_ood["ablate_probe"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "ablate_probe",
    InterventionHook("ablate", w_probe_hat), CFG,
    is_reasoning_model=True,
)

r1_ood["steer_-0.5"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "steer_-0.5",
    InterventionHook("steer", u_opp_steer, alpha=-0.5), CFG,
    is_reasoning_model=True,
)

r1_ood["steer_+0.2"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "steer_+0.2",
    InterventionHook("steer", u_opp_steer, alpha=0.2), CFG,
    is_reasoning_model=True,
)

r1_ood["steer_+0.3"] = run_batched(
    model_r1, tokenizer_r1, ood_tasks, "steer_+0.3",
    InterventionHook("steer", u_opp_steer, alpha=0.3), CFG,
    is_reasoning_model=True,
)

outfile_4a = Path(CFG.output_dir) / "phase4a_r1_ood_raw.json"
with open(outfile_4a, "w") as f:
    json.dump(r1_ood, f, indent=2, default=str)
print(f"\nSaved Phase 4A: {outfile_4a}")

r1_ood_metrics = {name: compute_metrics(res, name) for name, res in r1_ood.items()}
print_metrics(r1_ood_metrics, "PHASE 4A (R1-Distill OOD)")

mfile_4a = Path(CFG.output_dir) / "phase4a_metrics.json"
with open(mfile_4a, "w") as f:
    json.dump(r1_ood_metrics, f, indent=2)
print(f"Saved: {mfile_4a}")

print("\nUnloading R1-Distill to free VRAM...")
del model_r1, tokenizer_r1
gc.collect()
torch.cuda.empty_cache()
print(f"  GPU mem after unload: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# =====================================================================
# Phase 4B: Qwen-2.5-14B-Instruct (base) on OOD Tasks
# =====================================================================

print("\n" + "=" * 72)
print("  PHASE 4B: Qwen-2.5-14B-Instruct (base) on OOD Tasks")
print("=" * 72)

print("\nLoading base model...")
tokenizer_base = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
if tokenizer_base.pad_token is None:
    tokenizer_base.pad_token = tokenizer_base.eos_token
tokenizer_base.padding_side = "left"

model_base = AutoModelForCausalLM.from_pretrained(
    CFG.base_model, device_map="auto",
    torch_dtype=CFG.dtype, trust_remote_code=True,
)
model_base.eval()
print(f"  Loaded: {model_base.config.num_hidden_layers} layers")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

msgs = [{"role": "system", "content": CFG.system_prompt},
        {"role": "user", "content": ood_tasks[0]["prompt"]}]
test_text = tokenizer_base.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
test_inp = tokenizer_base(test_text, return_tensors="pt").to(model_base.device)
with torch.no_grad():
    test_out = model_base.generate(**test_inp, max_new_tokens=150, do_sample=False,
                                    pad_token_id=tokenizer_base.pad_token_id)
test_dec = tokenizer_base.decode(test_out[0][test_inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
print(f"  Sanity check: {test_dec[:150]}...")
assert len(test_dec) > 20, "Base model generation failed"
print("  OK\n")

base_ood = {}

base_ood["baseline"] = run_batched(
    model_base, tokenizer_base, ood_tasks, "baseline", None, CFG,
    is_reasoning_model=False,
)

base_ood["ablate_opp"] = run_batched(
    model_base, tokenizer_base, ood_tasks, "ablate_opp",
    InterventionHook("ablate", u_opp_hat), CFG,
    is_reasoning_model=False,
)

base_ood["ablate_random"] = run_batched(
    model_base, tokenizer_base, ood_tasks, "ablate_random",
    InterventionHook("ablate", u_rand_hat), CFG,
    is_reasoning_model=False,
)

base_ood["steer_+0.2"] = run_batched(
    model_base, tokenizer_base, ood_tasks, "steer_+0.2",
    InterventionHook("steer", u_opp_steer, alpha=0.2), CFG,
    is_reasoning_model=False,
)

base_ood["steer_+0.3"] = run_batched(
    model_base, tokenizer_base, ood_tasks, "steer_+0.3",
    InterventionHook("steer", u_opp_steer, alpha=0.3), CFG,
    is_reasoning_model=False,
)

outfile_4b = Path(CFG.output_dir) / "phase4b_base_ood_raw.json"
with open(outfile_4b, "w") as f:
    json.dump(base_ood, f, indent=2, default=str)
print(f"\nSaved Phase 4B: {outfile_4b}")

base_ood_metrics = {name: compute_metrics(res, name) for name, res in base_ood.items()}
print_metrics(base_ood_metrics, "PHASE 4B (Base OOD)")

mfile_4b = Path(CFG.output_dir) / "phase4b_metrics.json"
with open(mfile_4b, "w") as f:
    json.dump(base_ood_metrics, f, indent=2)
print(f"Saved: {mfile_4b}")


# =====================================================================
# Summary + Comparison
# =====================================================================

print("\n" + "=" * 72)
print("  PHASE 4 SUMMARY")
print("=" * 72)

def compare(label, r1_m, base_m, cond):
    r1 = r1_m.get(cond, {})
    base = base_m.get(cond, {})
    r1_v = r1.get(label, 0) if r1 else 0
    base_v = base.get(label, 0) if base else 0
    return r1_v, base_v

print(f"\n  Opp-mod lexical (mean/task):")
print(f"  {'Condition':<20} {'R1-Distill':>12} {'Base':>12}")
print(f"  {'-'*44}")
for cond in ["baseline", "ablate_opp", "ablate_random", "steer_+0.2", "steer_+0.3"]:
    r1_v = r1_ood_metrics.get(cond, {}).get("mean_opp_lex", 0)
    base_v = base_ood_metrics.get(cond, {}).get("mean_opp_lex", 0)
    print(f"  {cond:<20} {r1_v:>12.1f} {base_v:>12.1f}")

print(f"\n  Trigram repetition:")
print(f"  {'Condition':<20} {'R1-Distill':>12} {'Base':>12}")
print(f"  {'-'*44}")
for cond in ["baseline", "ablate_opp", "ablate_random", "steer_+0.2", "steer_+0.3"]:
    r1_v = r1_ood_metrics.get(cond, {}).get("mean_rep", 0)
    base_v = base_ood_metrics.get(cond, {}).get("mean_rep", 0)
    print(f"  {cond:<20} {r1_v:>12.3f} {base_v:>12.3f}")

print(f"\n  Mean output length (chars):")
print(f"  {'Condition':<20} {'R1-Distill':>12} {'Base':>12}")
print(f"  {'-'*44}")
for cond in ["baseline", "ablate_opp", "ablate_random", "steer_+0.2", "steer_+0.3"]:
    r1_v = r1_ood_metrics.get(cond, {}).get("mean_chars", 0)
    base_v = base_ood_metrics.get(cond, {}).get("mean_chars", 0)
    print(f"  {cond:<20} {r1_v:>12.0f} {base_v:>12.0f}")

print(f"\n\nGeneration complete.")
print(f"  Phase 4A: {outfile_4a}")
print(f"  Phase 4B: {outfile_4b}")
print(f"\nNext: run Phase 4 annotation cell.")
