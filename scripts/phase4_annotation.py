"""
Phase 4 Annotation — Chain-Mode
================================
Annotates generation outputs from Phase 4A (R1-Distill OOD) and Phase 4B
(Qwen-2.5-14B-Instruct OOD) using the same pipeline as Phase 1 / Phase 3.

Steps:
  1. Reads phase4a_r1_ood_raw.json and phase4b_base_ood_raw.json
  2. Segments each output using paragraph-split (same as Phase 1)
  3. Writes one segments JSONL per condition per sub-phase
  4. Calls annotate_chains.py on each → one API call per chain
  5. Computes per-condition label rates
  6. Prints a cross-model comparison table

Approximately 650 API calls (13 conditions × 50 tasks), ~$2–4, ~15 min.
No GPU required. Run in Colab top-to-bottom after Phase 4 generation.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter, defaultdict

# =====================================================================
# Config
# =====================================================================

PHASE4A_JSON = "/content/drive/MyDrive/workaround/phase0_toolkit/phase4_outputs/phase4a_r1_ood_raw.json"
PHASE4B_JSON = "/content/drive/MyDrive/workaround/phase0_toolkit/phase4_outputs/phase4b_base_ood_raw.json"
TOOLKIT_DIR  = "/content/drive/MyDrive/workaround/phase0_toolkit"
OUTPUT_DIR   = "/content/drive/MyDrive/workaround/phase0_toolkit/phase4_outputs/annotations"
GPT_MODEL    = "gpt-5.4"
CONCURRENCY  = 5

try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except:
    pass

sys.path.insert(0, TOOLKIT_DIR)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# =====================================================================
# Segmentation
# =====================================================================

def paragraph_split(text: str, max_chars: int = 1200) -> list:
    blocks = []
    for m in re.finditer(r'(?:(?!\n\s*\n).)+', text, re.DOTALL):
        block = m.group().strip()
        if block:
            blocks.append(block)

    segments = []
    for block in blocks:
        if len(block) <= max_chars:
            segments.append(block)
        else:
            start = 0
            while start < len(block):
                end = min(len(block), start + max_chars)
                cut = block.rfind("\n", start, end)
                if cut <= start:
                    cut = end
                chunk = block[start:cut].strip()
                if chunk:
                    segments.append(chunk)
                start = cut
    return segments


def make_segments_jsonl(
    phase_prefix: str,
    condition: str,
    results: list,
    out_dir: str,
    is_reasoning_model: bool,
) -> str:
    all_segments = []

    for record in results:
        task_id = f"{phase_prefix}__{condition}__{record['task_id']}"
        text = record.get("thinking", "") or record.get("full_output", "")

        if not text.strip():
            continue

        seg_texts = paragraph_split(text)
        for i, seg_text in enumerate(seg_texts):
            all_segments.append({
                "segment_id": f"{task_id}::seg{i:03d}",
                "task_id": task_id,
                "seg_index": i,
                "text": seg_text,
                "region": "thinking",
                "chain_truncated": record.get("truncated", False),
            })

    out_path = os.path.join(out_dir, f"{phase_prefix}_{condition}_segments.jsonl")
    with open(out_path, "w") as f:
        for seg in all_segments:
            f.write(json.dumps(seg) + "\n")

    return out_path


# =====================================================================
# Load + Segment
# =====================================================================

print("Loading Phase 4 outputs...")

phase4a_data = {}
phase4b_data = {}

if os.path.exists(PHASE4A_JSON):
    with open(PHASE4A_JSON) as f:
        phase4a_data = json.load(f)
    print(f"  Phase 4A (R1 OOD): {list(phase4a_data.keys())}")
else:
    print(f"  Phase 4A not found: {PHASE4A_JSON}")

if os.path.exists(PHASE4B_JSON):
    with open(PHASE4B_JSON) as f:
        phase4b_data = json.load(f)
    print(f"  Phase 4B (Base OOD): {list(phase4b_data.keys())}")
else:
    print(f"  Phase 4B not found: {PHASE4B_JSON}")

print("\nStep 1: Segmenting outputs...")
segment_files = {}

for condition, results in phase4a_data.items():
    if not results:
        continue
    seg_path = make_segments_jsonl("p4a", condition, results, OUTPUT_DIR,
                                   is_reasoning_model=True)
    n_segs = sum(1 for _ in open(seg_path))
    segment_files[("p4a", condition)] = seg_path
    print(f"  4A {condition}: {len(results)} chains -> {n_segs} segments")

for condition, results in phase4b_data.items():
    if not results:
        continue
    seg_path = make_segments_jsonl("p4b", condition, results, OUTPUT_DIR,
                                   is_reasoning_model=False)
    n_segs = sum(1 for _ in open(seg_path))
    segment_files[("p4b", condition)] = seg_path
    print(f"  4B {condition}: {len(results)} chains -> {n_segs} segments")


# =====================================================================
# Run annotate_chains.py on each condition
# =====================================================================

print("\nStep 2: Running chain-mode annotation...")
annotation_files = {}

for (phase, condition), seg_path in segment_files.items():
    ann_path = os.path.join(OUTPUT_DIR, f"{phase}_{condition}_labels.jsonl")

    if os.path.exists(ann_path):
        n_done = sum(1 for line in open(ann_path)
                     if '"ok"' in line or '"redacted_ok"' in line)
        n_segs = sum(1 for _ in open(seg_path))
        if n_done >= n_segs * 0.9:
            print(f"\n  {phase}/{condition}: already done ({n_done} labels), skipping")
            annotation_files[(phase, condition)] = ann_path
            continue

    print(f"\n  Annotating: {phase}/{condition}")
    cmd = [
        sys.executable,
        os.path.join(TOOLKIT_DIR, "annotate_chains.py"),
        "--segments", seg_path,
        "--output", ann_path,
        "--gpt_model", GPT_MODEL,
        "--concurrency", str(CONCURRENCY),
    ]
    print(f"    cmd: annotate_chains.py --segments {Path(seg_path).name} "
          f"--output {Path(ann_path).name} --gpt_model {GPT_MODEL}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        annotation_files[(phase, condition)] = ann_path
        n_labels = sum(1 for line in open(ann_path)
                       if '"ok"' in line or '"redacted_ok"' in line)
        print(f"    Done: {n_labels} segments labeled")
    else:
        print(f"    FAILED: {result.stderr[-500:]}")


# =====================================================================
# Compute label rates
# =====================================================================

print("\n\nStep 3: Computing label rates...")

ALL_LABELS = [
    "opponent_modeling", "iterated_reasoning", "equilibrium_identification",
    "payoff_analysis", "strategic_uncertainty", "cooperative_reasoning",
    "initialization", "deduction", "backtracking", "none_other",
]

SHORT = {
    "opponent_modeling": "opp_mod",
    "iterated_reasoning": "iter_rs",
    "equilibrium_identification": "eql_id",
    "payoff_analysis": "payoff",
    "strategic_uncertainty": "str_un",
    "cooperative_reasoning": "coop",
    "initialization": "init",
    "deduction": "deduct",
    "backtracking": "bktrak",
    "none_other": "none",
}


def get_label_rates(ann_path: str) -> dict:
    counter = Counter()
    n = 0
    with open(ann_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("status") not in ("ok", "redacted_ok", "ok_from_incomplete_chain"):
                continue
            n += 1
            for lab in rec.get("labels", []):
                counter[lab] += 1
    rates = {}
    for lab in ALL_LABELS:
        rates[lab] = counter.get(lab, 0) / max(1, n)
    return {"n_segments": n, "rates": rates, "counts": dict(counter)}


all_rates = {}
for key, ann_path in annotation_files.items():
    all_rates[key] = get_label_rates(ann_path)


# =====================================================================
# Print results
# =====================================================================

def print_phase_table(phase_prefix: str, phase_label: str, display_order: list):
    print(f"\n{'='*110}")
    print(f"  {phase_label} (chain-mode, {GPT_MODEL})")
    print(f"{'='*110}")

    header = f"  {'Condition':<20} {'Segs':>5}"
    for lab in ALL_LABELS:
        header += f" {SHORT[lab]:>7}"
    print(header)
    print(f"  {'-'*105}")

    for condition in display_order:
        key = (phase_prefix, condition)
        if key not in all_rates:
            continue
        r = all_rates[key]
        row = f"  {condition:<20} {r['n_segments']:>5}"
        for lab in ALL_LABELS:
            pct = 100 * r["rates"].get(lab, 0)
            row += f" {pct:>6.1f}%"
        print(row)


r1_order = ["baseline", "ablate_opp", "ablate_random", "ablate_payoff",
            "ablate_probe", "steer_-0.5", "steer_+0.2", "steer_+0.3"]
print_phase_table("p4a", "PHASE 4A: R1-Distill OOD", r1_order)

base_order = ["baseline", "ablate_opp", "ablate_random", "steer_+0.2", "steer_+0.3"]
print_phase_table("p4b", "PHASE 4B: Qwen-2.5-14B-Instruct (base) OOD", base_order)


# =====================================================================
# Cross-model comparison
# =====================================================================

print(f"\n{'='*110}")
print(f"  CROSS-MODEL COMPARISON (R1-Distill vs Base)")
print(f"{'='*110}")

shared_conds = ["baseline", "ablate_opp", "ablate_random", "steer_+0.2", "steer_+0.3"]
key_labels = ["opponent_modeling", "deduction", "payoff_analysis",
              "strategic_uncertainty", "backtracking", "initialization"]

for lab in key_labels:
    print(f"\n  {lab}:")
    print(f"  {'Condition':<20} {'R1-Distill':>12} {'Base':>12} {'Δ':>8}")
    print(f"  {'-'*52}")
    for cond in shared_conds:
        r1_key = ("p4a", cond)
        base_key = ("p4b", cond)
        r1_pct = 100 * all_rates.get(r1_key, {}).get("rates", {}).get(lab, 0)
        base_pct = 100 * all_rates.get(base_key, {}).get("rates", {}).get(lab, 0)
        delta = base_pct - r1_pct
        print(f"  {cond:<20} {r1_pct:>11.1f}% {base_pct:>11.1f}% {delta:>+7.1f}%")


# =====================================================================
# Key tests + save
# =====================================================================

print(f"\n{'='*72}")
print(f"  KEY TESTS")
print(f"{'='*72}")

print(f"\n  1. Ablation paradox (OOD):")
for phase, label in [("p4a", "R1-Distill"), ("p4b", "Base")]:
    b = all_rates.get((phase, "baseline"), {}).get("rates", {}).get("opponent_modeling", 0)
    a = all_rates.get((phase, "ablate_opp"), {}).get("rates", {}).get("opponent_modeling", 0)
    r = all_rates.get((phase, "ablate_random"), {}).get("rates", {}).get("opponent_modeling", 0)
    print(f"    {label}: baseline={100*b:.1f}%  ablate_opp={100*a:.1f}% (Δ={100*(a-b):+.1f}%)  "
          f"ablate_random={100*r:.1f}% (Δ={100*(r-b):+.1f}%)")

print(f"\n  2. Steering transfer (base model):")
for cond in ["baseline", "steer_+0.2", "steer_+0.3"]:
    key = ("p4b", cond)
    opp = 100 * all_rates.get(key, {}).get("rates", {}).get("opponent_modeling", 0)
    ded = 100 * all_rates.get(key, {}).get("rates", {}).get("deduction", 0)
    n = all_rates.get(key, {}).get("n_segments", 0)
    print(f"    {cond}: opp_mod={opp:.1f}%  deduction={ded:.1f}%  (N={n})")

print(f"\n  3. Content/control dissociation (steer_+0.3):")
for phase, label in [("p4a", "R1-Distill"), ("p4b", "Base")]:
    key = (phase, "steer_+0.3")
    r = all_rates.get(key, {})
    n = r.get("n_segments", 0)
    opp = 100 * r.get("rates", {}).get("opponent_modeling", 0)
    bkt = 100 * r.get("rates", {}).get("backtracking", 0)
    init = 100 * r.get("rates", {}).get("initialization", 0)
    print(f"    {label}: opp_mod={opp:.1f}%  backtrack={bkt:.1f}%  init={init:.1f}%  (N={n})")

report = {
    "model": GPT_MODEL,
    "phase4a": {cond: all_rates.get(("p4a", cond), {}) for cond in r1_order},
    "phase4b": {cond: all_rates.get(("p4b", cond), {}) for cond in base_order},
}
report_path = os.path.join(OUTPUT_DIR, "phase4_annotation_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved: {report_path}")
print("\nDone.")
