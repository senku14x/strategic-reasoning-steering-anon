"""
Phase 3: Chain-Mode Annotation of Intervention Outputs

What this script does:
  1. Reads phase3_raw_outputs_v3.json (all 8 conditions).
  2. Segments each output into paragraphs (same splitter as Phase 1).
  3. Writes one segments JSONL per condition.
  4. Calls annotate_chains.py on each condition (one API call per chain/task).
  5. Computes and prints per-condition label rates.
  6. Saves a summary report JSON.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# =====================================================================
# Config
# =====================================================================

PHASE3_JSON = "/content/drive/MyDrive/workaround/phase0_toolkit/phase3_outputs/phase3_raw_outputs_v3.json"
TOOLKIT_DIR = "/content/drive/MyDrive/workaround/phase0_toolkit"
OUTPUT_DIR  = "/content/drive/MyDrive/workaround/phase0_toolkit/phase3_outputs/annotations"
GPT_MODEL   = "gpt-5.4"
CONCURRENCY = 8

try:
    from google.colab import userdata
    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except Exception:
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


def make_segments_jsonl(condition: str, results: list, out_dir: str) -> str:
    all_segments = []
    for record in results:
        task_id = f"{condition}__{record['task_id']}"
        text = record.get("thinking", "") or record.get("full_output", "")
        if not text.strip():
            continue
        for i, seg_text in enumerate(paragraph_split(text)):
            all_segments.append({
                "segment_id":      f"{task_id}::seg{i:03d}",
                "task_id":         task_id,
                "seg_index":       i,
                "text":            seg_text,
                "region":          "thinking",
                "chain_truncated": record.get("truncated", False),
            })

    out_path = os.path.join(out_dir, f"{condition}_segments.jsonl")
    with open(out_path, "w") as f:
        for seg in all_segments:
            f.write(json.dumps(seg) + "\n")
    return out_path

# =====================================================================
# Step 1: Segment outputs
# =====================================================================

print("Loading Phase 3 outputs...")
with open(PHASE3_JSON) as f:
    phase3_data = json.load(f)
print(f"Conditions: {list(phase3_data.keys())}")

print("\nStep 1: Segmenting outputs...")
segment_files = {}
for condition, results in phase3_data.items():
    if not results:
        continue
    seg_path = make_segments_jsonl(condition, results, OUTPUT_DIR)
    n_segs = sum(1 for _ in open(seg_path))
    segment_files[condition] = seg_path
    print(f"  {condition}: {len(results)} chains → {n_segs} segments")

# =====================================================================
# Step 2: Annotate each condition
# =====================================================================

print("\nStep 2: Running chain-mode annotation...")
annotation_files = {}

for condition, seg_path in segment_files.items():
    ann_path = os.path.join(OUTPUT_DIR, f"{condition}_labels.jsonl")

    if os.path.exists(ann_path):
        n_done = sum(1 for line in open(ann_path) if '"ok"' in line or '"redacted_ok"' in line)
        n_segs = sum(1 for _ in open(seg_path))
        if n_done >= n_segs * 0.9:
            print(f"\n  {condition}: already done ({n_done} labels), skipping")
            annotation_files[condition] = ann_path
            continue

    print(f"\n  Annotating: {condition}")
    cmd = [
        sys.executable,
        os.path.join(TOOLKIT_DIR, "annotate_chains.py"),
        "--segments",    seg_path,
        "--output",      ann_path,
        "--gpt_model",   GPT_MODEL,
        "--concurrency", str(CONCURRENCY),
    ]
    print(f"    cmd: annotate_chains.py --segments {Path(seg_path).name} "
          f"--output {Path(ann_path).name} --gpt_model {GPT_MODEL}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        annotation_files[condition] = ann_path
        n_labels = sum(1 for line in open(ann_path) if '"ok"' in line or '"redacted_ok"' in line)
        print(f"    Done: {n_labels} segments labeled")
    else:
        print(f"    FAILED: {result.stderr[-300:]}")

# =====================================================================
# Step 3: Compute label rates
# =====================================================================

ALL_LABELS = [
    "opponent_modeling", "iterated_reasoning", "equilibrium_identification",
    "payoff_analysis", "strategic_uncertainty", "cooperative_reasoning",
    "initialization", "deduction", "backtracking", "none_other",
]

SHORT = {
    "opponent_modeling":        "opp_mod",
    "iterated_reasoning":       "iter_rs",
    "equilibrium_identification": "eql_id",
    "payoff_analysis":          "payoff",
    "strategic_uncertainty":    "str_un",
    "cooperative_reasoning":    "coop",
    "initialization":           "init",
    "deduction":                "deduct",
    "backtracking":             "bktrak",
    "none_other":               "none",
}

DISPLAY_ORDER = [
    "baseline", "ablate_opp", "ablate_random", "ablate_payoff",
    "ablate_probe", "steer_-0.5", "steer_+0.2", "steer_+0.3",
]


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
    return {
        "n_segments": n,
        "rates":  {lab: counter.get(lab, 0) / max(1, n) for lab in ALL_LABELS},
        "counts": dict(counter),
    }


print("\n\nStep 3: Computing label rates...")
all_rates = {condition: get_label_rates(ann_path)
             for condition, ann_path in annotation_files.items()}

# =====================================================================
# Step 4: Print results
# =====================================================================

print(f"\n{'='*110}")
print(f"  PHASE 3 ANNOTATION RESULTS (chain-mode, {GPT_MODEL})")
print(f"{'='*110}")

header = f"  {'Condition':<20} {'Segs':>5}"
for lab in ALL_LABELS:
    header += f" {SHORT[lab]:>7}"
print(header)
print(f"  {'-'*105}")

for condition in DISPLAY_ORDER:
    if condition not in all_rates:
        continue
    r = all_rates[condition]
    row = f"  {condition:<20} {r['n_segments']:>5}"
    for lab in ALL_LABELS:
        row += f" {100 * r['rates'].get(lab, 0):>6.1f}%"
    print(row)

if "baseline" in all_rates and "ablate_opp" in all_rates:
    print(f"\n  --- KEY COMPARISON ---")
    b = all_rates["baseline"]
    a = all_rates["ablate_opp"]
    for lab in ["opponent_modeling", "deduction", "payoff_analysis", "strategic_uncertainty"]:
        br = 100 * b["rates"].get(lab, 0)
        ar = 100 * a["rates"].get(lab, 0)
        print(f"  {lab:<25} baseline={br:.1f}%  ablate_opp={ar:.1f}%  Δ={ar-br:+.1f}%")

if "baseline" in all_rates and "steer_+0.2" in all_rates:
    print(f"\n  --- STEERING ---")
    b = all_rates["baseline"]
    for cond in ["steer_-0.5", "steer_+0.2", "steer_+0.3"]:
        if cond not in all_rates:
            continue
        s = all_rates[cond]
        opp_b = 100 * b["rates"].get("opponent_modeling", 0)
        opp_s = 100 * s["rates"].get("opponent_modeling", 0)
        ded_b = 100 * b["rates"].get("deduction", 0)
        ded_s = 100 * s["rates"].get("deduction", 0)
        print(f"  {cond:<15} opp_mod: {opp_b:.1f}%→{opp_s:.1f}% ({opp_s-opp_b:+.1f}%)  "
              f"deduction: {ded_b:.1f}%→{ded_s:.1f}% ({ded_s-ded_b:+.1f}%)")

# =====================================================================
# Step 5: Save report
# =====================================================================

report_path = os.path.join(OUTPUT_DIR, "phase3_annotation_report.json")
with open(report_path, "w") as f:
    json.dump({"model": GPT_MODEL, "conditions": all_rates}, f, indent=2)
print(f"\nSaved: {report_path}")
print("\nDone.")
