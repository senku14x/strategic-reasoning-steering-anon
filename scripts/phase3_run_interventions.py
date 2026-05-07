"""
Phase 3: Batched Ablation and Steering Interventions

Run this after:
  1. Phase 1 annotation is complete.
  2. Phase 2 activation extraction and DoM vector construction are complete.

After this script:
  Run scripts/phase3_annotate_outputs.py to obtain semantic label rates.

What this script does:
  1. Loads the dataset and selects 50 strategic tasks.
  2. Loads a pre-trained DeepSeek-R1-Distill-Qwen-14B model and tokenizer.
  3. Loads difference-of-means (DoM) steering vectors from Phase 2.
  4. Trains a logistic probe on opponent-modeling vs. deduction activations
     to extract a probe direction for comparison.
  5. Runs batched generation under 8 intervention conditions:
       - baseline         : no intervention (loaded from previous run if available)
       - ablate_opp       : project out the opponent-modeling DoM direction
       - ablate_random    : project out a random direction (specificity control)
       - ablate_payoff    : project out the payoff-analysis DoM direction (orthogonal control)
       - ablate_probe     : project out the logistic probe direction
       - steer_-0.5       : steer strongly against the opponent-modeling direction
       - steer_+0.2       : steer moderately toward the opponent-modeling direction
       - steer_+0.3       : steer more strongly toward the opponent-modeling direction
  6. Saves raw generation outputs to JSON.
  7. Computes judge-independent lexical metrics (output length, truncation,
     trigram repetition, opponent-modeling lexical hits, ground-truth substring hit rate).
  8. Prints a structured verdict comparing ablation vs. control effects.

"""

import gc
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

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
class Phase3Config:
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    system_prompt: str = (
        "You are a strategic reasoning assistant. Analyze the following "
        "problem carefully and explain your reasoning step by step before "
        "giving your answer."
    )

    dataset_file:    str = "/content/drive/MyDrive/workaround/final_dataset.json"
    chains_file:     str = "/content/drive/MyDrive/workaround/phase0_toolkit/r1_qwen14b_chains.json"
    vectors_npz:     str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_dom_vectors_v2.npz"
    activations_npz: str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_activations_v2.npz"
    baseline_json:   str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase3_outputs/phase3_raw_outputs.json"
    output_dir:      str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase3_outputs"

    max_new_tokens:     int = 3072
    batch_size:         int = 8
    intervention_layer: int = 24
    n_strategic_tasks:  int = 50
    n_control_tasks:    int = 50
    seed:               int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: object = torch.float16

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


CFG = Phase3Config()
print(f"\nConfig:")
print(f"  Batch size:     {CFG.batch_size}")
print(f"  max_new_tokens: {CFG.max_new_tokens}")
print(f"  Output:         {CFG.output_dir}")


# =====================================================================
# Data Loading
# =====================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tasks(config: Phase3Config):
    with open(config.dataset_file) as f:
        dataset = json.load(f)

    tasks = [
        {
            "task_id":       t["id"],
            "prompt":        t["task"],
            "category":      t.get("category", ""),
            "ground_truth":  t.get("ground_truth", ""),
            "optimal_action": t.get("optimal_action", ""),
        }
        for t in dataset["tasks"]
    ]

    strategic = [t for t in tasks if t["category"] != "non_strategic_control"]
    controls  = [t for t in tasks if t["category"] == "non_strategic_control"]

    rng = np.random.RandomState(config.seed)
    rng.shuffle(strategic)
    rng.shuffle(controls)

    strategic = strategic[:config.n_strategic_tasks]
    controls  = controls[:config.n_control_tasks]

    print(f"Loaded {len(dataset['tasks'])} tasks from {config.dataset_file}")
    print(f"  Strategic: {len(strategic)}, Controls: {len(controls)}")
    return strategic, controls


def load_baseline_for_tasks(config: Phase3Config, tasks: List[Dict]) -> Optional[List[Dict]]:
    path = Path(config.baseline_json)
    if not path.exists():
        print(f"  No saved baseline at {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    baseline = data.get("baseline", [])
    if not baseline:
        return None

    wanted_ids = [t["task_id"] for t in tasks]
    by_id      = {r["task_id"]: r for r in baseline}
    missing    = [tid for tid in wanted_ids if tid not in by_id]

    if missing:
        print(f"  Saved baseline missing {len(missing)} current task IDs; regenerating baseline.")
        return None

    aligned = [by_id[tid] for tid in wanted_ids]
    print(f"  Loaded aligned baseline for {len(aligned)} current tasks")
    return aligned


# =====================================================================
# Model + Vector Setup
# =====================================================================

strategic_tasks, _ = load_tasks(CFG)

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    CFG.model_name,
    device_map="auto",
    torch_dtype=CFG.dtype,
    trust_remote_code=True,
)
model.eval()
print(f"  Loaded: {model.config.num_hidden_layers} layers")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("\nLoading vectors...")
vec_npz = np.load(CFG.vectors_npz, allow_pickle=True)
wvw = {
    key.split("__")[1]: torch.from_numpy(vec_npz[key].astype(np.float32))
    for key in vec_npz.files
    if key.startswith("with_vs_without__")
}

L = CFG.intervention_layer
u_opp    = wvw["opponent_modeling"][L].to(CFG.device)
u_payoff = wvw["payoff_analysis"][L].to(CFG.device)

u_opp_hat    = u_opp / u_opp.norm()
u_payoff_hat = u_payoff / u_payoff.norm()

rng_np   = np.random.RandomState(CFG.seed)
rand_vec = torch.from_numpy(rng_np.randn(u_opp.shape[0]).astype(np.float32)).to(CFG.device)
u_rand_hat = rand_vec / rand_vec.norm()

MEAN_ACT_NORM = 158.6
u_opp_steer  = u_opp_hat * MEAN_ACT_NORM

print("Computing probe direction...")
act_data    = np.load(CFG.activations_npz, allow_pickle=True)
act_all     = act_data["activations"]
act_sids    = list(act_data["segment_ids"])
act_labels  = json.loads(str(act_data["labels_json"]))
act_regions = json.loads(str(act_data["regions_json"]))

think_idx  = [i for i, sid in enumerate(act_sids)
              if act_regions.get(sid, "thinking") == "thinking"]
think_acts = act_all[think_idx].astype(np.float32)
think_sids = [act_sids[i] for i in think_idx]

opp_m = np.array(["opponent_modeling" in act_labels.get(sid, []) for sid in think_sids])
ded_m = np.array(["deduction"         in act_labels.get(sid, []) for sid in think_sids])

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_p = np.concatenate([think_acts[opp_m & ~ded_m, L, :],
                      think_acts[ded_m & ~opp_m, L, :]], axis=0)
y_p = np.concatenate([np.ones((opp_m & ~ded_m).sum()),
                      np.zeros((ded_m & ~opp_m).sum())])

sc  = StandardScaler()
clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
clf.fit(sc.fit_transform(X_p), y_p)

w_probe     = clf.coef_[0] / sc.scale_
w_probe_hat = torch.from_numpy(
    (w_probe / np.linalg.norm(w_probe)).astype(np.float32)
).to(CFG.device)

del act_all, think_acts, X_p
gc.collect()
print(f"  cos(probe, u_opp) = "
      f"{F.cosine_similarity(w_probe_hat.unsqueeze(0), u_opp_hat.unsqueeze(0)).item():.4f}")

saved_baseline = load_baseline_for_tasks(CFG, strategic_tasks)

print("\n--- Sanity check ---")
msgs      = [{"role": "system", "content": CFG.system_prompt},
             {"role": "user",   "content": strategic_tasks[0]["prompt"]}]
test_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
test_inp  = tokenizer(test_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    test_out = model.generate(
        **test_inp, max_new_tokens=150, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
test_dec = tokenizer.decode(
    test_out[0][test_inp["input_ids"].shape[1]:], skip_special_tokens=True
)
print(f"  Output: {test_dec[:200]}...")
print(f"  Length: {len(test_dec)} chars")
assert len(test_dec) > 20, "Generation failed — check chat template"
print("  OK\n")


# =====================================================================
# Intervention Hook
# =====================================================================

class InterventionHook:
    def __init__(self, mode: str, direction: torch.Tensor, alpha: float = 1.0):
        self.mode      = mode
        self.direction = direction
        self.alpha     = alpha
        self.handle    = None

    def hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d = self.direction.to(hidden.device, dtype=hidden.dtype)

        if self.mode == "ablate":
            proj   = (hidden @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
            hidden = hidden - proj
        elif self.mode == "steer":
            hidden[:, -1, :] = hidden[:, -1, :] + self.alpha * d

        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def register(self, model, layer_idx: int):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[layer_idx]
        else:
            layer = model.transformer.h[layer_idx]
        self.handle = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# =====================================================================
# Generation
# =====================================================================

def format_prompt(task: Dict, system_prompt: str, tokenizer) -> str:
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": task["prompt"]}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_output(text: str) -> Dict:
    thinking, answer, truncated = "", text, False
    a = text.find("<think>")
    b = text.find("</think>")
    if a != -1 and b != -1 and b > a:
        thinking = text[a + len("<think>"):b].strip()
        answer   = text[b + len("</think>"):].strip()
    elif a != -1:
        thinking  = text[a + len("<think>"):].strip()
        answer    = ""
        truncated = True
    elif len(text) > 100:
        thinking = text
        answer   = ""
    return {"thinking": thinking, "answer": answer,
            "truncated": truncated, "full_output": text}


def run_batched(
    model,
    tokenizer,
    tasks: List[Dict],
    condition_name: str,
    intervention: Optional[InterventionHook],
    config: Phase3Config,
) -> List[Dict]:
    results  = []
    n        = len(tasks)
    bs       = config.batch_size
    n_batches = (n + bs - 1) // bs

    print(f"\n  Running: {condition_name} ({n} tasks, batch_size={bs}, {n_batches} batches)")

    if intervention:
        intervention.register(model, config.intervention_layer)

    try:
        t0 = time.time()
        for batch_start in tqdm(range(0, n, bs), desc=condition_name, total=n_batches):
            batch_tasks = tasks[batch_start:batch_start + bs]
            prompts     = [format_prompt(t, config.system_prompt, tokenizer) for t in batch_tasks]

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
                text       = tokenizer.decode(new_tokens, skip_special_tokens=True)
                parsed     = parse_output(text)
                real_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                results.append({
                    "task_id":       task["task_id"],
                    "category":      task["category"],
                    "condition":     condition_name,
                    "prompt":        task["prompt"],
                    "ground_truth":  task.get("ground_truth", ""),
                    "optimal_action": task.get("optimal_action", ""),
                    **parsed,
                    "n_tokens_padded_slice": len(new_tokens),
                    "n_real_tokens":         real_tokens,
                    "n_chars":               len(text),
                })

            gc.collect()
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        n_tok   = [r["n_tokens"] for r in results]
        n_trunc = sum(1 for r in results if r["truncated"])
        print(f"    Done: {len(results)} outputs in {elapsed:.0f}s "
              f"({elapsed/len(results):.1f}s/task), "
              f"mean_tok={np.mean(n_tok):.0f}, trunc={n_trunc}")
    finally:
        if intervention:
            intervention.remove()

    return results


# =====================================================================
# Run All Conditions
# =====================================================================

print("\n" + "=" * 72)
print("  PHASE 3: RUNNING ALL CONDITIONS")
print("=" * 72)

all_results = {}

if saved_baseline and len(saved_baseline) >= 50:
    all_results["baseline"] = saved_baseline
    print(f"\n  baseline: loaded {len(saved_baseline)} from previous run")
else:
    all_results["baseline"] = run_batched(
        model, tokenizer, strategic_tasks, "baseline", None, CFG,
    )

all_results["ablate_opp"] = run_batched(
    model, tokenizer, strategic_tasks, "ablate_opp",
    InterventionHook("ablate", u_opp_hat), CFG,
)

all_results["ablate_random"] = run_batched(
    model, tokenizer, strategic_tasks, "ablate_random",
    InterventionHook("ablate", u_rand_hat), CFG,
)

all_results["steer_+0.2"] = run_batched(
    model, tokenizer, strategic_tasks, "steer_+0.2",
    InterventionHook("steer", u_opp_steer, alpha=0.2), CFG,
)

all_results["ablate_probe"] = run_batched(
    model, tokenizer, strategic_tasks, "ablate_probe",
    InterventionHook("ablate", w_probe_hat), CFG,
)

all_results["steer_-0.5"] = run_batched(
    model, tokenizer, strategic_tasks, "steer_-0.5",
    InterventionHook("steer", u_opp_steer, alpha=-0.5), CFG,
)

all_results["steer_+0.3"] = run_batched(
    model, tokenizer, strategic_tasks, "steer_+0.3",
    InterventionHook("steer", u_opp_steer, alpha=0.3), CFG,
)

all_results["ablate_payoff"] = run_batched(
    model, tokenizer, strategic_tasks, "ablate_payoff",
    InterventionHook("ablate", u_payoff_hat), CFG,
)

print(f"\n  All conditions complete.")


# =====================================================================
# Save Raw Outputs
# =====================================================================

out_dir = Path(CFG.output_dir)
outfile = out_dir / "phase3_raw_outputs_v3.json"
with open(outfile, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
total = sum(len(v) for v in all_results.values())
print(f"\nSaved: {outfile}")
print(f"  Conditions:    {list(all_results.keys())}")
print(f"  Total outputs: {total}")


# =====================================================================
# Lexical Metrics
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

    chars  = [r["n_chars"]       for r in results]
    tokens = [r["n_real_tokens"] for r in results]
    n_trunc = sum(1 for r in results if r["truncated"])

    def trigram_rep(text: str) -> float:
        w = text.lower().split()
        if len(w) < 4:
            return 0.0
        trigrams = [tuple(w[i:i+3]) for i in range(len(w) - 2)]
        return 1.0 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0

    reps = [trigram_rep(r["full_output"]) for r in results]

    opp_lex = [
        len(OPP_RE.findall(r.get("thinking", "") or r.get("full_output", ""))) +
        len(BELIEF_RE.findall(r.get("thinking", "") or r.get("full_output", "")))
        for r in results
    ]

    n_correct = n_verif = 0
    for r in results:
        gt = str(r.get("ground_truth", "")).strip()
        if not gt:
            continue
        n_verif += 1
        ans = r.get("answer", "") or r.get("full_output", "")
        if gt.lower() in ans.lower():
            n_correct += 1

    n_think = sum(1 for r in results if "</think>" in r.get("full_output", ""))

    return {
        "condition":        name,
        "n":                n,
        "mean_chars":       float(np.mean(chars)),
        "mean_tokens":      float(np.mean(tokens)),
        "n_truncated":      n_trunc,
        "n_think_complete": n_think,
        "mean_rep":         float(np.mean(reps)),
        "mean_opp_lex":     float(np.mean(opp_lex)),
        "sum_opp_lex":      int(sum(opp_lex)),
        "n_verif":                    n_verif,
        "n_correct":                  n_correct,
        "ground_truth_substring_hit": float(n_correct / max(1, n_verif)),
    }


print("\n" + "=" * 72)
print("  METRICS")
print("=" * 72)

condition_order = [
    "baseline", "ablate_opp", "ablate_random", "ablate_payoff",
    "ablate_probe", "steer_-0.5", "steer_+0.2", "steer_+0.3",
]

all_metrics = {}
print(f"\n  {'Condition':<20} {'N':>3} {'Chars':>6} {'Tok':>5} "
      f"{'Trn':>3} {'ThkOK':>5} {'Rep':>5} {'OppLx':>5} {'GTHit':>5}")
print(f"  {'-'*65}")

for name in condition_order:
    if name not in all_results:
        continue
    m = compute_metrics(all_results[name], name)
    all_metrics[name] = m
    print(f"  {name:<20} {m['n']:>3} {m['mean_chars']:>6.0f} "
          f"{m['mean_tokens']:>5.0f} {m['n_truncated']:>3} "
          f"{m['n_think_complete']:>5} {m['mean_rep']:>5.3f} "
          f"{m['mean_opp_lex']:>5.1f} {m['ground_truth_substring_hit']:>5.2f}")


# =====================================================================
# Verdict
# =====================================================================

def print_verdict(m: Dict):
    print(f"\n{'='*72}")
    print(f"  PHASE 3 VERDICT (lexical proxies only)")
    print(f"{'='*72}")

    b = m.get("baseline", {})
    a = m.get("ablate_opp", {})
    r = m.get("ablate_random", {})
    p = m.get("ablate_probe", {})

    if not b or not a or b["n"] == 0:
        print("  Missing results.")
        return

    bl = b["mean_opp_lex"]
    al = a["mean_opp_lex"]
    rl = r.get("mean_opp_lex", bl)
    pl = p.get("mean_opp_lex", bl)

    delta_opp   = al - bl
    delta_rand  = rl - bl
    delta_probe = pl - bl

    print(f"\n  Opponent-modeling lexical proxy:")
    print(f"    baseline:       {bl:.1f}")
    print(f"    ablate_opp:     {al:.1f}  (Δ={delta_opp:+.1f})")
    print(f"    ablate_random:  {rl:.1f}  (Δ={delta_rand:+.1f})")
    print(f"    ablate_probe:   {pl:.1f}  (Δ={delta_probe:+.1f})")

    print(f"\n  Output length:")
    print(f"    baseline:       {b['mean_chars']:.0f}")
    print(f"    ablate_opp:     {a['mean_chars']:.0f}")
    print(f"    ablate_random:  {r.get('mean_chars', 0):.0f}")

    print(f"\n  Repetition:")
    print(f"    baseline:       {b['mean_rep']:.3f}")
    print(f"    ablate_opp:     {a['mean_rep']:.3f}")
    print(f"    ablate_random:  {r.get('mean_rep', 0):.3f}")

    steer_keys = sorted(k for k in m if k.startswith("steer_"))
    if steer_keys:
        print(f"\n  Steering:")
        for sk in steer_keys:
            sm = m[sk]
            print(f"    {sk}: opp_lex={sm['mean_opp_lex']:.1f}, "
                  f"chars={sm['mean_chars']:.0f}, rep={sm['mean_rep']:.3f}")

    print(f"\n  {'='*60}")
    if delta_opp > 0 and delta_opp > abs(delta_rand) * 2:
        print("  ABLATION PARADOX CANDIDATE:")
        print("    ablate_opp increases opponent-modeling lexical markers")
        print("    more than the random-direction ablation.")
        print("    Requires LLM/human annotation to interpret semantically.")
    elif abs(delta_opp) <= abs(delta_rand) * 1.5:
        print("  NON-SPECIFIC:")
        print("    ablate_opp is similar to random ablation on lexical proxies.")
    elif delta_opp < 0:
        print("  SUPPRESSION CANDIDATE:")
        print("    ablate_opp reduces opponent-modeling lexical markers.")
    else:
        print("  INCONCLUSIVE:")
        print(f"    Δ_opp={delta_opp:+.1f}, Δ_rand={delta_rand:+.1f}")
    print(f"  {'='*60}")
    print("\n  Note: lexical proxies are only triage metrics. Final conclusions use annotation.")


print_verdict(all_metrics)

mfile = Path(CFG.output_dir) / "phase3_metrics_v3.json"
with open(mfile, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nSaved: {mfile}")
print(f"Raw outputs: {outfile}")
print("\nDone. Next: annotate outputs with annotate_chains.py")
