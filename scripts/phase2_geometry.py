"""
Phase 2: Activation Extraction + Geometric Analysis (Multi-Label)
=================================================================
Extracts per-segment residual-stream activations from DeepSeek-R1-Distill-Qwen-14B
using mean pooling across token positions (paper Section 2.2). Computes difference-of-means
(DoM) vectors under 5 centering methods and runs the full robustness suite
(permutation, bootstrap, split-half).

Run on Google Colab with A100 high-RAM recommended for extraction.
Geometry analysis (Stage B) can re-run from saved NPZ without GPU.

Inputs:
    r1_qwen14b_chains.json              300 reasoning chains
    r1_qwen14b_segments.jsonl           14,645 paragraph-split segments
    r1_qwen14b_model_labels_chainmode.jsonl   Phase 1 multi-label annotations

Outputs:
    r1_qwen14b_activations_v2.npz       ~8-14 GB, (N, 48, 5120) fp16
    r1_qwen14b_dom_vectors_v2.npz       DoM vectors, 5 centering methods x 10 labels x 48 layers
    phase2_geometry_report.json          Full numerical results
    phase2_geometry_report.txt           Human-readable summary
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from collections import Counter, defaultdict
import gc
import time
import warnings
warnings.filterwarnings("ignore")

print("Imports OK")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


@dataclass
class Phase2Config:
    model_name:  str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    model_short: str = "r1_qwen14b"

    # UPDATE THESE PATHS
    chains_file:    str = "/content/drive/MyDrive/workaround/phase0_toolkit/r1_qwen14b_chains.json"
    segments_file:  str = "/content/drive/MyDrive/workaround/phase0_toolkit/r1_qwen14b_segments.jsonl"
    labels_file:    str = "/content/drive/MyDrive/workaround/phase0_toolkit/r1_qwen14b_model_labels_chainmode.jsonl"
    output_dir:     str = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs"

    max_length: int = 16384
    target_layers: Optional[List[int]] = None
    max_tasks: Optional[int] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: object = torch.float16

    headline_layer: int = 24
    n_bootstrap:  int = 1000
    n_permutation: int = 1000
    n_split_half:  int = 100
    seed: int = 42

    all_labels: List[str] = field(default_factory=lambda: [
        "opponent_modeling", "iterated_reasoning", "equilibrium_identification",
        "payoff_analysis", "strategic_uncertainty", "cooperative_reasoning",
        "initialization", "deduction", "backtracking", "none_other",
    ])
    strategic_labels: List[str] = field(default_factory=lambda: [
        "opponent_modeling", "iterated_reasoning", "strategic_uncertainty",
    ])
    analytical_labels: List[str] = field(default_factory=lambda: [
        "deduction", "payoff_analysis", "equilibrium_identification",
    ])

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


CFG = Phase2Config()
print(f"\nConfig:")
print(f"  Model:      {CFG.model_name}")
print(f"  Chains:     {CFG.chains_file}")
print(f"  Segments:   {CFG.segments_file}")
print(f"  Labels:     {CFG.labels_file}")
print(f"  max_length: {CFG.max_length}")
print(f"  Output:     {CFG.output_dir}")


# ── Data Loading ──────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_chains_by_task(path: str) -> Dict[str, str]:
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", data.get("tasks", []))
    by_task = {}
    for r in results:
        tid = r.get("task_id", "")
        fo = r.get("full_output", "")
        if tid and fo:
            by_task[tid] = fo
    print(f"Loaded {len(by_task)} chains from {path}")
    return by_task


def load_multilabel_annotations(path: str) -> Dict[str, List[str]]:
    seg_labels = {}
    for rec in load_jsonl(path):
        sid = rec.get("segment_id", "")
        labels = rec.get("labels", [])
        status = rec.get("status", "")
        if sid and labels and status in ("ok", "redacted_ok", "ok_from_incomplete_chain"):
            seg_labels[sid] = labels
    print(f"Loaded {len(seg_labels)} segment annotations from {path}")
    counter = Counter()
    for labels in seg_labels.values():
        for lab in labels:
            counter[lab] += 1
    multi_count = sum(1 for v in seg_labels.values() if len(v) > 1)
    print(f"  Multi-label segments: {multi_count} ({100*multi_count/max(1,len(seg_labels)):.1f}%)")
    print(f"  Label distribution:")
    for lab, n in counter.most_common():
        print(f"    {lab}: {n} ({100*n/max(1,len(seg_labels)):.1f}%)")
    return seg_labels


def build_segment_index(
    segments_jsonl: List[Dict],
    seg_labels: Dict[str, List[str]],
    full_outputs: Dict[str, str],
) -> List[Dict]:
    segments = []
    skipped_no_labels = skipped_no_chain = skipped_no_position = 0

    by_task = defaultdict(list)
    for seg in segments_jsonl:
        by_task[seg["task_id"]].append(seg)

    for tid, task_segs in by_task.items():
        full_output = full_outputs.get(tid)
        if not full_output:
            skipped_no_chain += len(task_segs)
            continue

        task_segs.sort(key=lambda s: s["seg_index"])
        search_start = 0
        for seg in task_segs:
            sid = seg["segment_id"]
            labels = seg_labels.get(sid)
            if not labels:
                skipped_no_labels += 1
                continue

            text = seg["text"]
            pos = full_output.find(text, search_start)
            if pos == -1:
                pos = full_output.find(text)
            if pos == -1:
                skipped_no_position += 1
                continue

            char_start = pos
            char_end = pos + len(text)
            search_start = char_end

            segments.append({
                "segment_id": sid, "task_id": tid,
                "seg_index": seg["seg_index"], "text": text,
                "char_start": char_start, "char_end": char_end,
                "labels": labels,
                "region": seg.get("region", "thinking"),
                "chain_truncated": seg.get("chain_truncated", False),
                "full_output": full_output,
            })

    print(f"\nSegment index built: {len(segments)} segments ready for extraction")
    print(f"  Skipped (no labels): {skipped_no_labels}, (no chain): {skipped_no_chain}, (no position): {skipped_no_position}")
    regions = Counter(s["region"] for s in segments)
    print(f"  Regions: {dict(regions)}")
    return segments


full_outputs  = load_chains_by_task(CFG.chains_file)
seg_labels    = load_multilabel_annotations(CFG.labels_file)
segments_raw  = load_jsonl(CFG.segments_file)
print(f"Loaded {len(segments_raw)} segments from {CFG.segments_file}")
segments = build_segment_index(segments_raw, seg_labels, full_outputs)


# ── Model Loading ─────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"\nLoading tokenizer: {CFG.model_name}")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model (dtype={CFG.dtype})...")
model = AutoModelForCausalLM.from_pretrained(
    CFG.model_name, device_map="auto",
    torch_dtype=CFG.dtype, trust_remote_code=True,
)
model.eval()
N_LAYERS  = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
print(f"  Loaded: {N_LAYERS} layers, hidden_dim={HIDDEN_DIM}")
print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ── Token Alignment ───────────────────────────────────────────────────

def char_to_token_range(char_start: int, char_end: int,
                        offset_mapping: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    token_start = token_end = None
    for idx, (cs, ce) in enumerate(offset_mapping):
        if cs == ce == 0:
            continue
        if token_start is None and ce > char_start:
            token_start = idx
        if cs < char_end:
            token_end = idx + 1
    if token_start is None or token_end is None or token_start >= token_end:
        return None
    return (token_start, token_end)


# ── Activation Extraction ─────────────────────────────────────────────

class ActivationCache:
    def __init__(self):
        self.activations = {}
        self.hooks = []

    def _hook(self, name):
        def fn(module, inp, out):
            self.activations[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    def register(self, model, layer_indices):
        self.clear()
        for li in layer_indices:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[li]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layer = model.transformer.h[li]
            else:
                raise ValueError("Cannot find layers in model architecture")
            self.hooks.append(layer.register_forward_hook(self._hook(f"L{li}")))

    def clear(self):
        self.activations = {}
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def stacked(self, layer_indices):
        acts = []
        for li in layer_indices:
            key = f"L{li}"
            if key in self.activations:
                acts.append(self.activations[key].squeeze(0))
        return torch.stack(acts, dim=0) if acts else None


def extract_one_chain(model, tokenizer, full_output, chain_segments, target_layers, max_length):
    enc = tokenizer(
        full_output, return_tensors="pt", return_offsets_mapping=True,
        add_special_tokens=False, truncation=True, max_length=max_length,
    )
    device = next(model.parameters()).device
    input_ids = enc.input_ids.to(device)
    offsets = enc.offset_mapping[0].tolist()
    seq_len = input_ids.shape[1]

    seg_token_ranges = {}
    for seg in chain_segments:
        tr = char_to_token_range(seg["char_start"], seg["char_end"], offsets)
        if tr is not None:
            ts, te = tr
            te = min(te, seq_len)
            if ts < te:
                seg_token_ranges[seg["segment_id"]] = (ts, te)

    if not seg_token_ranges:
        return {}

    cache = ActivationCache()
    cache.register(model, target_layers)
    try:
        with torch.no_grad():
            model(input_ids)
        acts = cache.stacked(target_layers)
        if acts is None:
            return {}

        results = {}
        for sid, (ts, te) in seg_token_ranges.items():
            seg_act = acts[:, ts:te, :].mean(dim=1)
            results[sid] = seg_act.cpu()
    finally:
        cache.clear()
    return results


def run_extraction(model, tokenizer, segments, config):
    layers = config.target_layers or list(range(N_LAYERS))
    by_chain = defaultdict(list)
    for seg in segments:
        by_chain[seg["task_id"]].append(seg)

    task_ids = sorted(by_chain.keys())
    if config.max_tasks:
        task_ids = task_ids[:config.max_tasks]

    all_activations = {}
    stats = {"chains_ok": 0, "chains_fail": 0, "segments_ok": 0, "segments_skip": 0}

    print(f"\nExtracting activations from {len(task_ids)} chains...")
    print(f"  Layers: {len(layers)} | max_length: {config.max_length}")

    for tid in tqdm(task_ids, desc="Chains"):
        chain_segs = by_chain[tid]
        full_output = chain_segs[0]["full_output"]
        try:
            chain_acts = extract_one_chain(
                model, tokenizer, full_output, chain_segs,
                target_layers=layers, max_length=config.max_length,
            )
            if chain_acts:
                stats["chains_ok"] += 1
                stats["segments_ok"] += len(chain_acts)
                stats["segments_skip"] += len(chain_segs) - len(chain_acts)
                all_activations.update(chain_acts)
            else:
                stats["chains_fail"] += 1
        except Exception as e:
            print(f"  ERROR on {tid}: {e}")
            stats["chains_fail"] += 1

        if stats["chains_ok"] % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nExtraction complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return all_activations


activations = run_extraction(model, tokenizer, segments, CFG)


# ── Save Raw Activations ─────────────────────────────────────────────

def save_activations(activations, segments, config):
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_meta = {s["segment_id"]: s for s in segments}
    ordered_ids = sorted(activations.keys())

    act_list, labels_dict, regions_dict = [], {}, {}
    for sid in ordered_ids:
        act_list.append(activations[sid].half().numpy())
        meta = seg_meta.get(sid, {})
        labels_dict[sid] = meta.get("labels", [])
        regions_dict[sid] = meta.get("region", "unknown")

    acts_array = np.stack(act_list, axis=0)
    outfile = out_dir / f"{config.model_short}_activations_v2.npz"
    print(f"\nSaving {acts_array.shape} activations to {outfile}...")
    np.savez_compressed(
        outfile, activations=acts_array,
        segment_ids=np.array(ordered_ids, dtype=object),
        labels_json=json.dumps(labels_dict),
        regions_json=json.dumps(regions_dict),
    )
    print(f"Saved: {outfile.stat().st_size / 1e9:.2f} GB")
    return outfile


activation_file = save_activations(activations, segments, CFG)

del model
gc.collect()
torch.cuda.empty_cache()
print("Model unloaded. Proceeding to geometry analysis.")


# ── Load from NPZ (to skip extraction) ───────────────────────────────

# Uncomment to skip extraction:
# npz = np.load("phase2_outputs/r1_qwen14b_activations_v2.npz", allow_pickle=True)
# acts = npz["activations"]
# sids = list(npz["segment_ids"])
# labels_dict = json.loads(str(npz["labels_json"]))
# regions_dict = json.loads(str(npz["regions_json"]))

ordered_ids = sorted(activations.keys())
seg_meta = {s["segment_id"]: s for s in segments}
acts = np.stack([activations[sid].half().numpy() for sid in ordered_ids], axis=0)
sids = ordered_ids
labels_dict = {sid: seg_meta[sid]["labels"] for sid in ordered_ids}
regions_dict = {sid: seg_meta[sid]["region"] for sid in ordered_ids}
print(f"\nReady for analysis: {acts.shape}")


# ── DoM Computation ───────────────────────────────────────────────────

def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_dom_vectors(acts, sids, labels_dict, regions_dict, config, think_only=True, min_count=20):
    mask = np.ones(len(sids), dtype=bool)
    if think_only:
        mask = np.array([regions_dict.get(sid, "thinking") == "thinking" for sid in sids])
    active_idx = np.where(mask)[0]
    active_acts = acts[active_idx].astype(np.float32)
    active_sids = [sids[i] for i in active_idx]
    n_seg, n_layers, hidden_dim = active_acts.shape
    print(f"\n  Segments for DoM: {n_seg} ({'think-only' if think_only else 'all'})")

    label_masks = {}
    for lab in config.all_labels:
        lab_mask = np.array([lab in labels_dict.get(sid, []) for sid in active_sids])
        label_masks[lab] = lab_mask
        print(f"    {lab}: {lab_mask.sum()} ({100*lab_mask.sum()/n_seg:.1f}%)")

    cooccur = Counter()
    for sid in active_sids:
        labs = tuple(sorted(labels_dict.get(sid, [])))
        if len(labs) > 1:
            for i in range(len(labs)):
                for j in range(i+1, len(labs)):
                    cooccur[(labs[i], labs[j])] += 1
    multi_n = sum(1 for sid in active_sids if len(labels_dict.get(sid, [])) > 1)
    print(f"\n  Multi-label segments: {multi_n} ({100*multi_n/n_seg:.1f}%)")

    h_global = active_acts.mean(axis=0)
    cat_means = {}
    for lab, lab_mask in label_masks.items():
        if lab_mask.sum() >= min_count:
            cat_means[lab] = active_acts[lab_mask].mean(axis=0)
    valid_labels = sorted(cat_means.keys())
    print(f"\n  Labels with >= {min_count} segments: {valid_labels}")

    print(f"\n  Category-mean vs h_global cosine (L{config.headline_layer}):")
    for lab in valid_labels:
        cm = cat_means[lab][config.headline_layer]
        gm = h_global[config.headline_layer]
        cos = float(np.dot(cm, gm) / (np.linalg.norm(cm) * np.linalg.norm(gm) + 1e-10))
        print(f"    {lab}: {cos:.6f}")

    # Method 1: With-vs-without (primary for multi-label)
    wvw = {}
    for lab in valid_labels:
        lm = label_masks[lab]
        wvw[lab] = active_acts[lm].mean(axis=0) - active_acts[~lm].mean(axis=0)

    # Method 2: Original (h_c - h_global)
    orig = {lab: cat_means[lab] - h_global for lab in valid_labels}

    # Method 3: Leave-one-out
    loo = {}
    total_sum = active_acts.sum(axis=0)
    for lab in valid_labels:
        lm = label_masks[lab]
        n_c = lm.sum()
        remaining_n = n_seg - n_c
        if remaining_n > 0:
            h_global_exc = (total_sum - active_acts[lm].sum(axis=0)) / remaining_n
            loo[lab] = cat_means[lab] - h_global_exc

    # Method 4: Class-balanced
    balanced_global = np.stack(list(cat_means.values()), axis=0).mean(axis=0)
    cb = {lab: cat_means[lab] - balanced_global for lab in valid_labels}

    return {
        "with_vs_without": wvw, "original": orig, "leave_one_out": loo,
        "class_balanced": cb, "raw_means": cat_means, "h_global": h_global,
        "valid_labels": valid_labels, "label_masks": label_masks,
        "active_acts": active_acts, "active_sids": active_sids,
        "co_occurrence": dict(cooccur), "multi_label_count": multi_n, "n_segments": n_seg,
    }


dom_results = compute_dom_vectors(acts, sids, labels_dict, regions_dict, CFG)


# ── Geometry Analysis ─────────────────────────────────────────────────

def compute_geometry(dom_results, config):
    L = config.headline_layer
    valid = dom_results["valid_labels"]
    results = {}

    for method_name in ["with_vs_without", "original", "leave_one_out", "class_balanced"]:
        vecs = dom_results[method_name]
        labels_in = [lab for lab in valid if lab in vecs]
        n = len(labels_in)
        matrix = np.zeros((n, n))
        for i, c1 in enumerate(labels_in):
            for j, c2 in enumerate(labels_in):
                matrix[i, j] = 1.0 if i == j else cosine_sim(vecs[c1][L], vecs[c2][L])
        results[f"cosine_matrix_{method_name}"] = {"labels": labels_in, "matrix": matrix.tolist()}

        if "opponent_modeling" in labels_in and "deduction" in labels_in:
            i_om = labels_in.index("opponent_modeling")
            i_ded = labels_in.index("deduction")
            print(f"\n  {method_name} (L{L}): opp_mod vs deduction = {matrix[i_om, i_ded]:+.4f}")

    # Pairwise raw cosine
    raw = dom_results["raw_means"]
    key_pairs = [
        ("opponent_modeling", "deduction"), ("opponent_modeling", "payoff_analysis"),
        ("opponent_modeling", "iterated_reasoning"), ("opponent_modeling", "strategic_uncertainty"),
    ]
    pairwise = {}
    for c1, c2 in key_pairs:
        if c1 in raw and c2 in raw:
            val = cosine_sim(raw[c1][L], raw[c2][L])
            pairwise[f"{c1}_vs_{c2}"] = val
            print(f"  Pairwise raw (L{L}): {c1} vs {c2} = {val:+.6f}")
    results["pairwise_raw"] = pairwise

    # Layer-wise analysis
    n_layers = dom_results["with_vs_without"][valid[0]].shape[0]
    for method_name in ["with_vs_without", "original", "leave_one_out", "class_balanced"]:
        vecs = dom_results[method_name]
        if "opponent_modeling" in vecs and "deduction" in vecs:
            layerwise = [cosine_sim(vecs["opponent_modeling"][l], vecs["deduction"][l]) for l in range(n_layers)]
            results[f"layerwise_{method_name}"] = layerwise
            arr = np.array(layerwise)
            print(f"\n  {method_name} layer-wise opp_mod vs ded: Min={arr.min():.4f} Max={arr.max():.4f} Mean={arr.mean():.4f} Neg={(arr<0).sum()}/{n_layers}")

    # SVD preview
    wvw = dom_results["with_vs_without"]
    svd_labels = [lab for lab in valid if lab in wvw]
    if len(svd_labels) >= 3:
        mat = np.stack([wvw[lab][L] for lab in svd_labels], axis=0)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        total_var = (S**2).sum()
        explained = (S**2) / total_var
        results["svd_singular_values"] = S.tolist()
        results["svd_explained_ratio"] = explained.tolist()
        results["svd_labels"] = svd_labels
        print(f"\n  SVD at L{L}:")
        for k in range(min(5, len(S))):
            print(f"    SV{k+1}: {S[k]:.2f} ({100*explained[k]:.1f}%)")

    return results


geometry = compute_geometry(dom_results, CFG)


# ── Robustness Suite ──────────────────────────────────────────────────

def run_robustness_suite(dom_results, config):
    L = config.headline_layer
    acts_L = dom_results["active_acts"][:, L, :].copy().astype(np.float32)
    label_masks = dom_results["label_masks"]
    rng = np.random.RandomState(config.seed)
    N = acts_L.shape[0]
    results = {}

    opp_mask = label_masks.get("opponent_modeling")
    ded_mask = label_masks.get("deduction")
    if opp_mask is None or ded_mask is None:
        return results

    def wvw_cosine(mask_c, acts_layer):
        return acts_layer[mask_c].mean(axis=0) - acts_layer[~mask_c].mean(axis=0)

    v_opp = wvw_cosine(opp_mask, acts_L)
    v_ded = wvw_cosine(ded_mask, acts_L)
    observed = cosine_sim(v_opp, v_ded)
    print(f"\n  Observed (with-vs-without, L{L}): {observed:+.4f}")

    # Permutation test
    print(f"  Running permutation test (n={config.n_permutation})...")
    n_opp, n_ded = opp_mask.sum(), ded_mask.sum()
    null_cosines = []
    for _ in range(config.n_permutation):
        perm_opp = np.zeros(N, dtype=bool)
        perm_opp[rng.choice(N, size=n_opp, replace=False)] = True
        perm_ded = np.zeros(N, dtype=bool)
        perm_ded[rng.choice(N, size=n_ded, replace=False)] = True
        null_cosines.append(cosine_sim(wvw_cosine(perm_opp, acts_L), wvw_cosine(perm_ded, acts_L)))
    null_cosines = np.array(null_cosines)
    p_val = float(np.mean(null_cosines <= observed))
    print(f"    p={p_val:.6f}, null mean={null_cosines.mean():.4f}")
    results["permutation"] = {"observed": observed, "p_value": p_val, "null_mean": float(null_cosines.mean()), "null_std": float(null_cosines.std()), "null_min": float(null_cosines.min()), "null_max": float(null_cosines.max())}

    # Bootstrap
    print(f"  Running bootstrap (n={config.n_bootstrap})...")
    boot_cosines = []
    for _ in range(config.n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        b_acts = acts_L[idx]
        b_opp, b_ded = opp_mask[idx], ded_mask[idx]
        if b_opp.sum() < 5 or b_ded.sum() < 5:
            continue
        boot_cosines.append(cosine_sim(wvw_cosine(b_opp, b_acts), wvw_cosine(b_ded, b_acts)))
    boot_cosines = np.array(boot_cosines)
    ci_lo, ci_hi = float(np.percentile(boot_cosines, 2.5)), float(np.percentile(boot_cosines, 97.5))
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    results["bootstrap"] = {"ci_lo": ci_lo, "ci_hi": ci_hi, "mean": float(boot_cosines.mean()), "std": float(boot_cosines.std())}

    # Split-half
    print(f"  Running split-half (n={config.n_split_half})...")
    split_cosines = []
    for _ in range(config.n_split_half):
        perm = rng.permutation(N)
        for half_idx in [perm[:N//2], perm[N//2:]]:
            h_acts = acts_L[half_idx]
            h_opp, h_ded = opp_mask[half_idx], ded_mask[half_idx]
            if h_opp.sum() < 5 or h_ded.sum() < 5:
                continue
            split_cosines.append(cosine_sim(wvw_cosine(h_opp, h_acts), wvw_cosine(h_ded, h_acts)))
    split_cosines = np.array(split_cosines)
    n_neg = (split_cosines < 0).sum()
    print(f"    Mean={split_cosines.mean():.4f}, std={split_cosines.std():.4f}, negative={n_neg}/{len(split_cosines)}")
    results["split_half"] = {"mean": float(split_cosines.mean()), "std": float(split_cosines.std()), "n_negative": int(n_neg), "n_total": len(split_cosines)}

    return results


robustness = run_robustness_suite(dom_results, CFG)


# ── Verdict ───────────────────────────────────────────────────────────

def print_verdict(geometry, robustness, config):
    L = config.headline_layer
    print("\n" + "=" * 72)
    print("  PHASE 2 GO/NO-GO VERDICT")
    print("=" * 72)

    key = "cosine_matrix_with_vs_without"
    if key in geometry:
        labels = geometry[key]["labels"]
        mat = np.array(geometry[key]["matrix"])
        if "opponent_modeling" in labels and "deduction" in labels:
            val = mat[labels.index("opponent_modeling"), labels.index("deduction")]
            print(f"\n  Primary (with-vs-without, L{L}): {val:+.4f}")
            if val <= -0.6:
                print(f"  STRONG GO: Antagonism survives. Proceed to Phase 2.5.")
            elif val <= -0.3:
                print(f"  WEAK GO: Antagonism real but weaker. Proceed with caution.")
            else:
                print(f"  NO-GO: Antagonism was an annotation artifact. Branch 3.")

    if "bootstrap" in robustness:
        bs = robustness["bootstrap"]
        print(f"  Bootstrap 95% CI: [{bs['ci_lo']:.4f}, {bs['ci_hi']:.4f}]")
    if "permutation" in robustness:
        print(f"  Permutation p-value: {robustness['permutation']['p_value']:.6f}")


print_verdict(geometry, robustness, CFG)


# ── Save Results ──────────────────────────────────────────────────────

out_dir = Path(CFG.output_dir)

vec_dict = {}
for method in ["with_vs_without", "original", "leave_one_out", "class_balanced"]:
    for lab, v in dom_results[method].items():
        vec_dict[f"{method}__{lab}"] = v.astype(np.float16)
vec_file = out_dir / f"{CFG.model_short}_dom_vectors_v2.npz"
np.savez_compressed(vec_file, **vec_dict)
print(f"Saved vectors: {vec_file}")

report = {
    "config": {"model": CFG.model_name, "headline_layer": CFG.headline_layer, "n_segments": dom_results["n_segments"], "multi_label_count": dom_results["multi_label_count"]},
    "geometry": {k: v for k, v in geometry.items()},
    "robustness": robustness,
}
json_file = out_dir / "phase2_geometry_report.json"
with open(json_file, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"Saved report: {json_file}")
print("\nPhase 2 complete.")
