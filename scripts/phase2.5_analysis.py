"""
Phase 2 + Phase 2.5 Full Analysis
==================================
Loads phase2 activation and DoM vector NPZ outputs, then runs:

  Phase 2 Analysis (Cells 1-3):
    - Per-layer cosine tables for antagonism pairs, ToM cluster alignment,
      and other key pairs
    - Full 10x10 cosine matrix at layers 0, 12, 24, 36, 47
    - Layer-wise norm profiles

  Phase 2.5 (Cells 4-10):
    Three complementary tests to determine if the opp_mod/deduction
    antagonism is a single signed axis or two separable directions:
      Test 1 — Linear probe vs DoM direction agreement
      Test 2 — SVD depth profile across all 48 layers
      Test 3 — Co-occurrence geometry (segments labeled BOTH)
    Integrated verdict synthesis.

  Phase 2.5 Extended (Cells 11-18):
    Investigates WHY probe and DoM direction disagree (cos=0.15):
      - Probe accuracy and probe-DoM cosine at every layer
      - Regularization sweep (does stronger reg recover DoM direction?)
      - Probe direction decomposed onto DoM vectors and SVD basis
      - Within-class variance and Fisher discriminant comparison
      - 2D scatter of segments on top SVD components
      - Probe direction stability across layers

No GPU needed — pure numpy/scipy/sklearn.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

print("Imports OK")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACTIVATIONS_NPZ = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_activations_v2.npz"
VECTORS_NPZ     = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/r1_qwen14b_dom_vectors_v2.npz"
REPORT_JSON     = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs/phase2_geometry_report.json"
OUTPUT_DIR      = "/content/drive/MyDrive/workaround/phase0_toolkit/phase2_outputs"

HEADLINE_LAYER = 24
N_LAYERS = 48
HIDDEN_DIM = 5120

ALL_LABELS = [
    "opponent_modeling", "iterated_reasoning", "equilibrium_identification",
    "payoff_analysis", "strategic_uncertainty", "cooperative_reasoning",
    "initialization", "deduction", "backtracking", "none_other",
]

SHORT = {
    "opponent_modeling": "opp-mod",
    "iterated_reasoning": "iter-reas",
    "equilibrium_identification": "equil-id",
    "payoff_analysis": "payoff",
    "strategic_uncertainty": "strat-unc",
    "cooperative_reasoning": "coop-reas",
    "initialization": "init",
    "deduction": "deduction",
    "backtracking": "backtrack",
    "none_other": "none/other",
}

# ---------------------------------------------------------------------------
# Load activations
# ---------------------------------------------------------------------------

print(f"Loading activations: {ACTIVATIONS_NPZ}")
npz = np.load(ACTIVATIONS_NPZ, allow_pickle=True)
acts_all     = npz["activations"]
segment_ids  = list(npz["segment_ids"])
labels_dict  = json.loads(str(npz["labels_json"]))
regions_dict = json.loads(str(npz["regions_json"]))
print(f"  Shape: {acts_all.shape}, segments: {len(segment_ids)}")

think_mask = np.array([regions_dict.get(sid, "thinking") == "thinking"
                       for sid in segment_ids])
acts = acts_all[think_mask].astype(np.float32)
sids = [segment_ids[i] for i in range(len(segment_ids)) if think_mask[i]]
N_SEG = acts.shape[0]
print(f"  Think-only: {N_SEG} segments")

label_masks = {}
for lab in ALL_LABELS:
    mask = np.array([lab in labels_dict.get(sid, []) for sid in sids])
    label_masks[lab] = mask
    n = mask.sum()
    if n > 0:
        print(f"    {lab}: {n}")

# ---------------------------------------------------------------------------
# Load DoM vectors
# ---------------------------------------------------------------------------

print(f"\nLoading vectors: {VECTORS_NPZ}")
vec_npz = np.load(VECTORS_NPZ, allow_pickle=True)
wvw_vectors = {}
for lab in ALL_LABELS:
    key = f"with_vs_without__{lab}"
    if key in vec_npz:
        wvw_vectors[lab] = vec_npz[key].astype(np.float32)
print(f"  Loaded {len(wvw_vectors)} with-vs-without vectors")

with open(REPORT_JSON) as f:
    report = json.load(f)
print(f"  Report loaded")

# ---------------------------------------------------------------------------
# Pre-compute shared masks
# ---------------------------------------------------------------------------

opp_mask  = label_masks["opponent_modeling"]
ded_mask  = label_masks["deduction"]
opp_only  = opp_mask & ~ded_mask
ded_only  = ded_mask & ~opp_mask
both_mask = opp_mask & ded_mask

idx_opp = np.where(opp_only)[0]
idx_ded = np.where(ded_only)[0]

print(f"\nReady: {N_SEG} think-only segments")
print(f"  opp_only={len(idx_opp)}, ded_only={len(idx_ded)}, both={both_mask.sum()}")


# ===========================================================================
# Helpers
# ===========================================================================

def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ===========================================================================
# Phase 2 — Cell 1: Per-Layer Cosine Table
# ===========================================================================

def print_layerwise_table(wvw_vectors, pairs, every_n=1):
    pair_names = [f"{SHORT[c1]} vs {SHORT[c2]}" for c1, c2 in pairs]
    header = f"{'Layer':>6}" + "".join(f"{p:>18}" for p in pair_names)
    print(header)
    print("-" * len(header))
    for layer in range(N_LAYERS):
        if layer % every_n != 0:
            continue
        row = f"{layer:>6}"
        for c1, c2 in pairs:
            if c1 in wvw_vectors and c2 in wvw_vectors:
                val = cosine_sim(wvw_vectors[c1][layer], wvw_vectors[c2][layer])
                row += f"{val:>+18.4f}"
            else:
                row += f"{'N/A':>18}"
        marker = "  <-- L24" if layer == HEADLINE_LAYER else ""
        print(row + marker)


print("\n" + "=" * 90)
print("  LAYER-WISE COSINE: ANTAGONISM PAIRS (with-vs-without DoM)")
print("=" * 90)
antagonism_pairs = [
    ("opponent_modeling", "deduction"),
    ("strategic_uncertainty", "deduction"),
    ("iterated_reasoning", "deduction"),
    ("cooperative_reasoning", "deduction"),
]
print_layerwise_table(wvw_vectors, antagonism_pairs)

print("\n" + "=" * 90)
print("  LAYER-WISE COSINE: ToM CLUSTER ALIGNMENT (with-vs-without DoM)")
print("=" * 90)
alignment_pairs = [
    ("opponent_modeling", "strategic_uncertainty"),
    ("opponent_modeling", "iterated_reasoning"),
    ("opponent_modeling", "cooperative_reasoning"),
    ("strategic_uncertainty", "iterated_reasoning"),
]
print_layerwise_table(wvw_vectors, alignment_pairs)

print("\n" + "=" * 90)
print("  LAYER-WISE COSINE: OTHER KEY PAIRS (with-vs-without DoM)")
print("=" * 90)
other_pairs = [
    ("opponent_modeling", "payoff_analysis"),
    ("deduction", "payoff_analysis"),
    ("none_other", "payoff_analysis"),
    ("none_other", "opponent_modeling"),
]
print_layerwise_table(wvw_vectors, other_pairs)


# ===========================================================================
# Phase 2 — Cell 2: Full Cosine Matrix at Multiple Layers
# ===========================================================================

def print_cosine_matrix_at_layer(wvw_vectors, layer, labels=None):
    if labels is None:
        labels = [l for l in ALL_LABELS if l in wvw_vectors]
    short_names = [SHORT.get(l, l[:8]) for l in labels]
    print(f"\n  Full cosine matrix at Layer {layer}:")
    header = f"{'':>12}" + "".join(f"{s:>12}" for s in short_names)
    print(header)
    print("-" * len(header))
    for i, c1 in enumerate(labels):
        row = f"{short_names[i]:>12}"
        for j, c2 in enumerate(labels):
            if i == j:
                row += f"{'+1.000':>12}"
            else:
                val = cosine_sim(wvw_vectors[c1][layer], wvw_vectors[c2][layer])
                row += f"{val:>+12.3f}"
        print(row)


for layer in [0, 12, 24, 36, 47]:
    print(f"\n{'='*90}")
    print_cosine_matrix_at_layer(wvw_vectors, layer)


# ===========================================================================
# Phase 2.5 — Cell 3: Test 1 — Linear Probe vs DoM Direction
# ===========================================================================
#
# Train logistic regression to separate opp_mod-only from deduction-only
# segments (hold out BOTH). Compare learned weight vector w to
# (u_opp - u_ded) via cosine. High agreement (>0.9) means probe and DoM
# find the same discriminative direction; low agreement means the DoM
# contrast is not the primary separator. Held-out BOTH segments near the
# decision boundary support a single-axis interpretation.

def test_probe_vs_dom(acts, sids, label_masks, wvw_vectors, layer):
    print(f"\n{'='*72}")
    print(f"  TEST 1: LINEAR PROBE vs DoM DIRECTION (Layer {layer})")
    print(f"{'='*72}")

    n_opp_only = opp_only.sum()
    n_ded_only = ded_only.sum()
    n_both     = both_mask.sum()
    print(f"\n  opp_mod only:   {n_opp_only}")
    print(f"  deduction only: {n_ded_only}")
    print(f"  BOTH:           {n_both} (held out)")

    X = np.concatenate([acts[idx_opp, layer, :], acts[idx_ded, layer, :]], axis=0)
    y = np.concatenate([np.ones(len(idx_opp)), np.zeros(len(idx_ded))], axis=0)
    print(f"  Training set: {len(X)} samples ({len(idx_opp)} opp, {len(idx_ded)} ded)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")
    print(f"\n  5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    clf.fit(X_scaled, y)
    w_unscaled = clf.coef_[0] / scaler.scale_

    u_opp = wvw_vectors["opponent_modeling"][layer]
    u_ded = wvw_vectors["deduction"][layer]
    dom_direction = u_opp - u_ded

    cos_w_dom = cosine_sim(w_unscaled, dom_direction)
    cos_w_opp = cosine_sim(w_unscaled, u_opp)
    cos_w_ded = cosine_sim(w_unscaled, u_ded)

    print(f"\n  Probe weight vector (w) cosine with:")
    print(f"    u_opp - u_ded (DoM contrast): {cos_w_dom:+.4f}")
    print(f"    u_opp (opp-mod direction):     {cos_w_opp:+.4f}")
    print(f"    u_ded (deduction direction):   {cos_w_ded:+.4f}")

    print(f"\n  Interpretation:")
    if abs(cos_w_dom) > 0.9:
        print(f"    HIGH AGREEMENT ({cos_w_dom:+.4f}): Probe and DoM find the same direction.")
        print(f"    Consistent with a single discriminative axis.")
    elif abs(cos_w_dom) > 0.6:
        print(f"    MODERATE AGREEMENT ({cos_w_dom:+.4f}): Probe finds a related but not")
        print(f"    identical direction. Some of the variance comes from elsewhere.")
    else:
        print(f"    LOW AGREEMENT ({cos_w_dom:+.4f}): Probe finds a different discriminative")
        print(f"    direction. The DoM contrast is NOT the primary separator.")

    probs = None
    if n_both > 0:
        X_both = acts[both_mask, layer, :]
        X_both_scaled = scaler.transform(X_both)
        probs = clf.predict_proba(X_both_scaled)[:, 1]
        print(f"\n  Held-out BOTH segments ({n_both}):")
        print(f"    Mean P(opp_mod): {probs.mean():.4f}")
        print(f"    Std:             {probs.std():.4f}")
        print(f"    Classified as opp: {(probs > 0.5).sum()}, as ded: {(probs <= 0.5).sum()}")
        if abs(probs.mean() - 0.5) < 0.1:
            print(f"    → BOTH segments sit near decision boundary (mean ~0.5).")
            print(f"      Consistent with single-axis: carrying both labels → near zero projection.")
        else:
            direction = "opp_mod" if probs.mean() > 0.5 else "deduction"
            print(f"    → BOTH segments lean toward {direction} (mean={probs.mean():.3f}).")
            print(f"      Suggests asymmetric co-occurrence, not clean cancellation.")

    return {
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cos_w_dom_contrast": cos_w_dom,
        "cos_w_opp": cos_w_opp,
        "cos_w_ded": cos_w_ded,
        "n_opp_only": int(n_opp_only),
        "n_ded_only": int(n_ded_only),
        "n_both": int(n_both),
        "both_mean_prob": float(probs.mean()) if probs is not None else None,
    }


probe_result = test_probe_vs_dom(acts, sids, label_masks, wvw_vectors, HEADLINE_LAYER)


# ===========================================================================
# Phase 2.5 — Cell 4: Test 2 — SVD Depth Profile
# ===========================================================================
#
# Run SVD of the category-mean matrix at every layer. Track how the
# singular value spectrum changes with depth. If top-1 dominance
# increases in middle layers the geometry becomes more single-axis where
# the model is doing the most work.

def test_svd_depth_profile(wvw_vectors, layers=None):
    print(f"\n{'='*72}")
    print(f"  TEST 2: SVD DEPTH PROFILE")
    print(f"{'='*72}")

    if layers is None:
        layers = list(range(N_LAYERS))

    labels_for_svd = [l for l in ALL_LABELS if l in wvw_vectors
                      and label_masks[l].sum() >= 20]

    top1_pct = []
    top2_pct = []
    top3_pct = []

    for layer in layers:
        mat = np.stack([wvw_vectors[lab][layer] for lab in labels_for_svd], axis=0)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        total_var = (S**2).sum()
        explained = (S**2) / total_var
        top1_pct.append(explained[0])
        top2_pct.append(explained[0] + explained[1])
        top3_pct.append(explained[0] + explained[1] + explained[2])

    top1_pct = np.array(top1_pct)
    top2_pct = np.array(top2_pct)
    top3_pct = np.array(top3_pct)

    print(f"\n  Labels used ({len(labels_for_svd)}): "
          f"{[SHORT[l] for l in labels_for_svd]}")

    print(f"\n  {'Layer':>6}  {'SV1%':>8}  {'SV1+2%':>8}  {'SV1+2+3%':>10}  {'Verdict':>12}")
    print(f"  {'-'*50}")
    for layer in layers:
        v1 = 100 * top1_pct[layer]
        v2 = 100 * top2_pct[layer]
        v3 = 100 * top3_pct[layer]
        if v1 > 70:
            verdict = "SINGLE-AXIS"
        elif v2 > 85:
            verdict = "~2-axis"
        else:
            verdict = "multi-dim"
        marker = "  <--" if layer == HEADLINE_LAYER else ""
        print(f"  {layer:>6}  {v1:>7.1f}%  {v2:>7.1f}%  {v3:>9.1f}%  {verdict:>12}{marker}")

    print(f"\n  Summary across all layers:")
    print(f"    SV1 range:   [{100*top1_pct.min():.1f}%, {100*top1_pct.max():.1f}%]")
    print(f"    SV1+2 range: [{100*top2_pct.min():.1f}%, {100*top2_pct.max():.1f}%]")
    print(f"    Layers where SV1 > 70%: {(top1_pct > 0.7).sum()}/{N_LAYERS}")
    print(f"    Layers where SV1+2 > 85%: {(top2_pct > 0.85).sum()}/{N_LAYERS}")

    mean_sv1 = top1_pct.mean()
    mean_sv2 = top2_pct.mean()
    print(f"\n  Mean SV1: {100*mean_sv1:.1f}%, Mean SV1+2: {100*mean_sv2:.1f}%")
    if mean_sv1 > 0.6:
        print(f"  → Geometry is predominantly single-axis across depth.")
    elif mean_sv2 > 0.75:
        print(f"  → Geometry is predominantly two-dimensional across depth.")
    else:
        print(f"  → Geometry is genuinely multi-dimensional (3+ axes) across depth.")

    return {
        "top1_pct": top1_pct.tolist(),
        "top2_pct": top2_pct.tolist(),
        "top3_pct": top3_pct.tolist(),
        "labels": labels_for_svd,
    }


svd_result = test_svd_depth_profile(wvw_vectors)


# ===========================================================================
# Phase 2.5 — Cell 5: Test 3 — Co-Occurrence Geometry
# ===========================================================================
#
# Segments labeled with BOTH opp_mod AND deduction. If single axis, their
# projection onto u_opp should be near zero. If two separable circuits,
# their projection onto u_opp should be positive AND onto u_ded should
# also be positive (both circuits active simultaneously).

def test_cooccurrence_geometry(acts, sids, label_masks, wvw_vectors, layer):
    print(f"\n{'='*72}")
    print(f"  TEST 3: CO-OCCURRENCE GEOMETRY (Layer {layer})")
    print(f"{'='*72}")

    n_both    = both_mask.sum()
    n_opp     = opp_only.sum()
    n_ded     = ded_only.sum()
    n_neither = (~opp_mask & ~ded_mask).sum()

    print(f"\n  Segment counts:")
    print(f"    opp_mod only:  {n_opp}")
    print(f"    deduction only: {n_ded}")
    print(f"    BOTH:           {n_both}")
    print(f"    Neither:        {n_neither}")

    if n_both < 5:
        print(f"\n  Too few BOTH segments ({n_both}) for meaningful analysis.")
        return {}

    u_opp = wvw_vectors["opponent_modeling"][layer]
    u_ded = wvw_vectors["deduction"][layer]
    u_opp_hat = u_opp / (np.linalg.norm(u_opp) + 1e-10)
    u_ded_hat = u_ded / (np.linalg.norm(u_ded) + 1e-10)
    u_contrast = u_opp - u_ded
    u_contrast_hat = u_contrast / (np.linalg.norm(u_contrast) + 1e-10)

    h_global = acts[:, layer, :].mean(axis=0)

    groups = {
        "opp_only": opp_only,
        "ded_only": ded_only,
        "BOTH": both_mask,
        "neither": ~opp_mask & ~ded_mask,
    }

    print(f"\n  Projections onto DoM directions (centered, then dot with unit vec):")
    print(f"  {'Group':>12}  {'N':>6}  {'proj(u_opp)':>14}  {'proj(u_ded)':>14}  {'proj(u_opp-u_ded)':>18}")
    print(f"  {'-'*70}")

    results = {}
    for name, mask in groups.items():
        if mask.sum() < 2:
            continue
        group_acts = acts[mask, layer, :] - h_global[np.newaxis, :]
        proj_opp = group_acts @ u_opp_hat
        proj_ded = group_acts @ u_ded_hat
        proj_contrast = group_acts @ u_contrast_hat
        m_opp = proj_opp.mean()
        m_ded = proj_ded.mean()
        m_con = proj_contrast.mean()
        print(f"  {name:>12}  {mask.sum():>6}  {m_opp:>+14.4f}  {m_ded:>+14.4f}  {m_con:>+18.4f}")
        results[name] = {
            "n": int(mask.sum()),
            "proj_opp_mean": float(m_opp),
            "proj_ded_mean": float(m_ded),
            "proj_contrast_mean": float(m_con),
            "proj_opp_std": float(proj_opp.std()),
            "proj_ded_std": float(proj_ded.std()),
        }

    print(f"\n  Interpretation:")
    both_data = results.get("BOTH", {})
    opp_data  = results.get("opp_only", {})
    ded_data  = results.get("ded_only", {})

    if both_data:
        both_contrast = both_data["proj_contrast_mean"]
        opp_contrast  = opp_data.get("proj_contrast_mean", 0)
        ded_contrast  = ded_data.get("proj_contrast_mean", 0)

        print(f"\n    On the opp-ded contrast axis (u_opp - u_ded):")
        print(f"      opp_only:   {opp_contrast:+.4f}  (should be positive)")
        print(f"      ded_only:   {ded_contrast:+.4f}  (should be negative)")
        print(f"      BOTH:       {both_contrast:+.4f}")

        midpoint  = (opp_contrast + ded_contrast) / 2
        range_val = abs(opp_contrast - ded_contrast)
        relative_pos = (both_contrast - midpoint) / (range_val / 2) if range_val > 0 else 0

        print(f"\n      BOTH relative position on contrast axis: {relative_pos:+.3f}")
        print(f"        (-1 = at ded_only, 0 = midpoint, +1 = at opp_only)")

        if abs(relative_pos) < 0.3:
            print(f"\n    → BOTH segments sit near the midpoint of the contrast axis.")
            print(f"      CONSISTENT WITH SINGLE-AXIS: carrying both labels → cancellation.")
        else:
            lean = "opp_mod" if relative_pos > 0 else "deduction"
            print(f"\n    → BOTH segments lean toward {lean} (pos={relative_pos:+.3f}).")
            print(f"      NOT consistent with simple cancellation on a single axis.")

        both_proj_opp = both_data["proj_opp_mean"]
        both_proj_ded = both_data["proj_ded_mean"]
        print(f"\n    On individual directions:")
        print(f"      BOTH proj(u_opp) = {both_proj_opp:+.4f}  "
              f"(opp_only: {opp_data.get('proj_opp_mean', 0):+.4f})")
        print(f"      BOTH proj(u_ded) = {both_proj_ded:+.4f}  "
              f"(ded_only: {ded_data.get('proj_ded_mean', 0):+.4f})")

        if both_proj_opp > 0 and both_proj_ded > 0:
            print(f"\n    → BOTH segments project positively on BOTH directions.")
            print(f"      CONSISTENT WITH TWO-AXIS: both circuits active simultaneously.")
        elif both_proj_opp > 0 and both_proj_ded < 0:
            print(f"\n    → Positive on u_opp, negative on u_ded. Opp-mod dominates.")
        elif both_proj_opp < 0 and both_proj_ded > 0:
            print(f"\n    → Negative on u_opp, positive on u_ded. Deduction dominates.")
        else:
            print(f"\n    → Both projections negative — unexpected for dual-labeled segments.")

    return results


cooccur_result = test_cooccurrence_geometry(
    acts, sids, label_masks, wvw_vectors, HEADLINE_LAYER
)


# ===========================================================================
# Phase 2.5 — Cell 6: Integrated Verdict
# ===========================================================================

def phase25_verdict(probe_result, svd_result, cooccur_result):
    print(f"\n{'='*72}")
    print(f"  PHASE 2.5 INTEGRATED VERDICT")
    print(f"{'='*72}")

    cos_wd = probe_result["cos_w_dom_contrast"]
    cv_acc = probe_result["cv_accuracy"]
    print(f"\n  Test 1 (Probe vs DoM):")
    print(f"    cos(w, u_opp - u_ded) = {cos_wd:+.4f}")
    print(f"    CV accuracy = {cv_acc:.4f}")

    if abs(cos_wd) > 0.9:
        t1_verdict = "single-axis"
        print(f"    → Probe agrees with DoM contrast direction: SINGLE-AXIS signal")
    elif abs(cos_wd) > 0.6:
        t1_verdict = "ambiguous"
        print(f"    → Moderate agreement: AMBIGUOUS")
    else:
        t1_verdict = "multi-axis"
        print(f"    → Probe finds different direction: MULTI-AXIS")

    top1_mean   = np.mean(svd_result["top1_pct"])
    top2_mean   = np.mean(svd_result["top2_pct"])
    top1_at_L24 = svd_result["top1_pct"][HEADLINE_LAYER]
    print(f"\n  Test 2 (SVD):")
    print(f"    SV1 at L24:  {100*top1_at_L24:.1f}%")
    print(f"    Mean SV1:    {100*top1_mean:.1f}%")
    print(f"    Mean SV1+2:  {100*top2_mean:.1f}%")

    if top1_mean > 0.6:
        t2_verdict = "single-axis"
        print(f"    → Dominant first component: SINGLE-AXIS")
    elif top2_mean > 0.75:
        t2_verdict = "two-axis"
        print(f"    → Two comparable components: TWO-AXIS")
    else:
        t2_verdict = "multi-axis"
        print(f"    → Spread across 3+ components: MULTI-AXIS")

    both_data = cooccur_result.get("BOTH", {})
    print(f"\n  Test 3 (Co-occurrence):")
    if both_data and both_data["n"] >= 5:
        both_proj_opp = both_data["proj_opp_mean"]
        both_proj_ded = both_data["proj_ded_mean"]
        contrast_val  = both_data["proj_contrast_mean"]
        print(f"    BOTH proj(u_opp) = {both_proj_opp:+.4f}")
        print(f"    BOTH proj(u_ded) = {both_proj_ded:+.4f}")
        print(f"    BOTH proj(contrast) = {contrast_val:+.4f}")

        if abs(contrast_val) < 0.5 * abs(cooccur_result.get("opp_only", {}).get("proj_contrast_mean", 1)):
            t3_verdict = "single-axis"
            print(f"    → Near-zero on contrast: CONSISTENT WITH SINGLE-AXIS")
        elif both_proj_opp > 0 and both_proj_ded > 0:
            t3_verdict = "two-axis"
            print(f"    → Positive on both: CONSISTENT WITH TWO-AXIS")
        else:
            t3_verdict = "ambiguous"
            print(f"    → Mixed signal: AMBIGUOUS")
    else:
        t3_verdict = "insufficient"
        print(f"    → Too few BOTH segments for reliable test")

    verdicts = [t1_verdict, t2_verdict, t3_verdict]
    print(f"\n  {'='*60}")
    print(f"  SYNTHESIS:")
    print(f"    Test 1 (Probe):    {t1_verdict}")
    print(f"    Test 2 (SVD):      {t2_verdict}")
    print(f"    Test 3 (Co-occur): {t3_verdict}")

    if verdicts.count("single-axis") >= 2:
        print(f"\n  → SINGLE-AXIS STORY.")
        print(f"    Social–analytical is encoded as a single signed dimension.")
        print(f"    Phase 3 ablation should break BOTH opp-mod and deduction.")
        print(f"    Alternative-interpretation test (Phase 6.5) becomes mandatory.")
    elif verdicts.count("two-axis") >= 2:
        print(f"\n  → TWO-AXIS STORY.")
        print(f"    Keep antagonism framing with nuance. Phase 3 ablation should")
        print(f"    break opp-mod specifically, NOT deduction.")
    elif verdicts.count("multi-axis") >= 2:
        print(f"\n  → MULTI-DIMENSIONAL STORY.")
        print(f"    The geometry is richer than one or two axes. Consider whether the")
        print(f"    antagonism narrative is the right framing.")
    else:
        print(f"\n  → MIXED/AMBIGUOUS.")
        print(f"    Tests disagree. Proceed to Phase 3 but do not commit to either")
        print(f"    framing yet.")
    print(f"  {'='*60}")

    return {
        "test1_verdict": t1_verdict,
        "test2_verdict": t2_verdict,
        "test3_verdict": t3_verdict,
        "verdicts": verdicts,
    }


verdict_25 = phase25_verdict(probe_result, svd_result, cooccur_result)


# ===========================================================================
# Phase 2.5 Extended — Cell 7: Probe at All 48 Layers
# ===========================================================================
#
# For each layer train logistic regression and record CV accuracy and
# cosine(w, u_opp - u_ded). This tells us where the model most separates
# the two categories and whether the probe-DoM disagreement is universal
# or layer-specific.

def probe_all_layers(acts, idx_opp, idx_ded, wvw_vectors, C=1.0):
    print(f"\n{'='*72}")
    print(f"  PROBE AT ALL 48 LAYERS (C={C})")
    print(f"{'='*72}")

    results = []
    print(f"\n  {'Layer':>6}  {'CV Acc':>8}  {'cos(w, DoM)':>12}  "
          f"{'cos(w,u_opp)':>13}  {'cos(w,u_ded)':>13}  {'|w| along DoM':>14}")
    print(f"  {'-'*72}")

    for layer in range(N_LAYERS):
        X = np.concatenate([acts[idx_opp, layer, :],
                            acts[idx_ded, layer, :]], axis=0)
        y = np.concatenate([np.ones(len(idx_opp)),
                            np.zeros(len(idx_ded))], axis=0)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", random_state=42)
        cv = cross_val_score(clf, X_s, y, cv=3, scoring="accuracy")
        acc = cv.mean()
        clf.fit(X_s, y)
        w_unscaled = clf.coef_[0] / scaler.scale_

        u_opp = wvw_vectors["opponent_modeling"][layer]
        u_ded = wvw_vectors["deduction"][layer]
        dom_contrast = u_opp - u_ded

        cos_dom = cosine_sim(w_unscaled, dom_contrast)
        cos_opp = cosine_sim(w_unscaled, u_opp)
        cos_ded = cosine_sim(w_unscaled, u_ded)
        dom_hat = dom_contrast / (np.linalg.norm(dom_contrast) + 1e-10)
        proj_magnitude = abs(np.dot(w_unscaled, dom_hat)) / (np.linalg.norm(w_unscaled) + 1e-10)

        marker = "  <--" if layer == HEADLINE_LAYER else ""
        print(f"  {layer:>6}  {acc:>8.4f}  {cos_dom:>+12.4f}  "
              f"{cos_opp:>+13.4f}  {cos_ded:>+13.4f}  {proj_magnitude:>14.4f}{marker}")

        results.append({
            "layer": layer,
            "cv_accuracy": float(acc),
            "cos_w_dom": cos_dom,
            "cos_w_opp": cos_opp,
            "cos_w_ded": cos_ded,
            "proj_along_dom": float(proj_magnitude),
            "w_unscaled": w_unscaled,
        })

    accs     = np.array([r["cv_accuracy"] for r in results])
    cos_doms = np.array([r["cos_w_dom"] for r in results])
    projs    = np.array([r["proj_along_dom"] for r in results])

    best_acc_layer = int(accs.argmax())
    best_cos_layer = int(np.abs(cos_doms).argmax())

    print(f"\n  Summary:")
    print(f"    Best CV accuracy:     {accs.max():.4f} at L{best_acc_layer}")
    print(f"    Worst CV accuracy:    {accs.min():.4f} at L{int(accs.argmin())}")
    print(f"    Highest |cos(w,DoM)|: {np.abs(cos_doms).max():.4f} at L{best_cos_layer}")
    print(f"    Mean |cos(w,DoM)|:    {np.abs(cos_doms).mean():.4f}")
    print(f"    Mean proj along DoM:  {projs.mean():.4f}")
    print(f"    Layers where |cos(w,DoM)| > 0.5: {(np.abs(cos_doms) > 0.5).sum()}/48")
    print(f"    Layers where |cos(w,DoM)| > 0.3: {(np.abs(cos_doms) > 0.3).sum()}/48")

    return results


probe_results = probe_all_layers(acts, idx_opp, idx_ded, wvw_vectors, C=1.0)


# ===========================================================================
# Phase 2.5 Extended — Cell 8: Regularization Sweep
# ===========================================================================
#
# With weak regularization (high C) the probe can exploit any direction in
# 5120-D space. With strong regularization (low C) it is forced toward
# high-SNR directions. If the DoM direction has the highest SNR, strong
# regularization should push cos(w, DoM) upward. If it doesn't, the DoM
# direction genuinely isn't the best discriminative direction even in the
# SNR sense.

def regularization_sweep(acts, idx_opp, idx_ded, wvw_vectors, layer=24):
    print(f"\n{'='*72}")
    print(f"  REGULARIZATION SWEEP (Layer {layer})")
    print(f"  Does stronger regularization push probe toward DoM?")
    print(f"{'='*72}")

    X = np.concatenate([acts[idx_opp, layer, :],
                        acts[idx_ded, layer, :]], axis=0)
    y = np.concatenate([np.ones(len(idx_opp)),
                        np.zeros(len(idx_ded))], axis=0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    u_opp = wvw_vectors["opponent_modeling"][layer]
    u_ded = wvw_vectors["deduction"][layer]
    dom_contrast = u_opp - u_ded

    C_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

    print(f"\n  {'C':>10}  {'CV Acc':>8}  {'cos(w, DoM)':>12}  "
          f"{'|w| along DoM':>14}  {'w L2 norm':>10}")
    print(f"  {'-'*60}")

    results = []
    for C in C_values:
        clf = LogisticRegression(max_iter=3000, C=C, solver="lbfgs", random_state=42)
        cv = cross_val_score(clf, X_s, y, cv=5, scoring="accuracy")
        clf.fit(X_s, y)
        w_unscaled = clf.coef_[0] / scaler.scale_
        w_norm = np.linalg.norm(w_unscaled)
        cos_dom = cosine_sim(w_unscaled, dom_contrast)
        dom_hat = dom_contrast / (np.linalg.norm(dom_contrast) + 1e-10)
        proj = abs(np.dot(w_unscaled, dom_hat)) / (w_norm + 1e-10)
        print(f"  {C:>10.4f}  {cv.mean():>8.4f}  {cos_dom:>+12.4f}  "
              f"{proj:>14.4f}  {w_norm:>10.2f}")
        results.append({
            "C": C,
            "cv_accuracy": float(cv.mean()),
            "cos_w_dom": cos_dom,
            "proj_along_dom": float(proj),
            "w_norm": float(w_norm),
        })

    cos_at_strong  = results[0]["cos_w_dom"]
    cos_at_weak    = results[-1]["cos_w_dom"]
    cos_at_default = [r for r in results if r["C"] == 1.0][0]["cos_w_dom"]

    print(f"\n  Interpretation:")
    print(f"    cos(w,DoM) at C=0.0001 (strongest reg): {cos_at_strong:+.4f}")
    print(f"    cos(w,DoM) at C=1.0    (default):       {cos_at_default:+.4f}")
    print(f"    cos(w,DoM) at C=100    (weakest reg):   {cos_at_weak:+.4f}")

    if abs(cos_at_strong) > abs(cos_at_default) + 0.15:
        print(f"\n    → Strong regularization INCREASES agreement with DoM.")
        print(f"      The DoM direction IS a high-SNR direction, but the")
        print(f"      unregularized probe finds higher-dimensional features.")
    elif abs(cos_at_strong) < 0.3 and abs(cos_at_default) < 0.3:
        print(f"\n    → Agreement stays low regardless of regularization.")
        print(f"      The DoM direction is genuinely NOT the best discriminative")
        print(f"      direction at any regularization strength.")
    else:
        print(f"\n    → Mixed pattern. Examine the trend above.")

    return results


reg_results = regularization_sweep(acts, idx_opp, idx_ded, wvw_vectors, HEADLINE_LAYER)


# ===========================================================================
# Phase 2.5 Extended — Cell 9: What IS the Probe Direction?
# ===========================================================================
#
# Project the probe weight vector onto each individual DoM vector (which
# categories does the probe most align with?) and onto the top SVD
# components of the category-mean matrix (where does the probe live in
# the low-dimensional structure?).

def analyze_probe_direction(acts, idx_opp, idx_ded, wvw_vectors, layer=24):
    print(f"\n{'='*72}")
    print(f"  WHAT IS THE PROBE DIRECTION? (Layer {layer})")
    print(f"{'='*72}")

    X = np.concatenate([acts[idx_opp, layer, :],
                        acts[idx_ded, layer, :]], axis=0)
    y = np.concatenate([np.ones(len(idx_opp)),
                        np.zeros(len(idx_ded))], axis=0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_s, y)
    w = clf.coef_[0] / scaler.scale_
    w_hat = w / (np.linalg.norm(w) + 1e-10)

    print(f"\n  Cosine of probe direction with each DoM vector:")
    print(f"  {'Label':>20}  {'cos(w, u_label)':>16}  {'Interpretation':>20}")
    print(f"  {'-'*60}")

    cos_with_labels = {}
    for lab in ALL_LABELS:
        if lab in wvw_vectors:
            c = cosine_sim(w, wvw_vectors[lab][layer])
            cos_with_labels[lab] = c
            if abs(c) > 0.3:
                interp = "strong align" if c > 0 else "strong anti-align"
            elif abs(c) > 0.15:
                interp = "weak align" if c > 0 else "weak anti-align"
            else:
                interp = "orthogonal"
            print(f"  {SHORT[lab]:>20}  {c:>+16.4f}  {interp:>20}")

    sorted_cos = sorted(cos_with_labels.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Probe direction most aligns with (positive = more opp-mod-like):")
    for lab, c in sorted_cos[:3]:
        print(f"    {SHORT[lab]}: {c:+.4f}")
    print(f"  Probe direction most anti-aligns with:")
    for lab, c in sorted_cos[-3:]:
        print(f"    {SHORT[lab]}: {c:+.4f}")

    svd_labels = [l for l in ALL_LABELS if l in wvw_vectors and label_masks[l].sum() >= 20]
    mat = np.stack([wvw_vectors[lab][layer] for lab in svd_labels], axis=0)
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

    print(f"\n  Probe direction projected onto SVD components of DoM matrix:")
    print(f"  {'Component':>12}  {'Var explained':>14}  {'Projection':>12}")
    print(f"  {'-'*42}")
    total_proj = 0
    for k in range(min(6, len(S))):
        sv_direction = Vt[k]
        proj = np.dot(w_hat, sv_direction)
        var_exp = (S[k]**2) / (S**2).sum()
        total_proj += proj**2
        print(f"  {'SV'+str(k+1):>12}  {100*var_exp:>13.1f}%  {proj:>+12.4f}")

    print(f"\n  Total variance of w in top-6 SVD subspace: {100*total_proj:.1f}%")
    print(f"  Remaining (orthogonal to DoM structure):   {100*(1-total_proj):.1f}%")

    if total_proj < 0.3:
        print(f"\n  → The probe direction is mostly ORTHOGONAL to the DoM structure.")
        print(f"    It's using dimensions that the category-mean analysis doesn't see.")
        print(f"    This likely means within-class variance structure (not means)")
        print(f"    drives classification.")
    elif total_proj > 0.7:
        print(f"\n  → The probe direction lives within the DoM structure.")
        print(f"    The disagreement is about WHICH combination of DoM components,")
        print(f"    not about using a totally different subspace.")
    else:
        print(f"\n  → The probe direction is partly in the DoM structure ({100*total_proj:.0f}%)")
        print(f"    and partly orthogonal. Mixed signal.")

    return {"cos_with_labels": cos_with_labels, "svd_projection_total": float(total_proj)}


probe_dir_result = analyze_probe_direction(
    acts, idx_opp, idx_ded, wvw_vectors, HEADLINE_LAYER
)


# ===========================================================================
# Phase 2.5 Extended — Cell 10: Within-Class Variance Analysis
# ===========================================================================
#
# If deduction has much higher within-class variance than opp-mod, the DoM
# direction (which captures mean difference) will be a poor classifier
# because deduction segments are spread over a large volume. The probe
# would then find directions that cut through the low-variance subspace of
# deduction, which could be orthogonal to DoM. Fisher's discriminant ratio
# is computed along both the DoM contrast direction and the probe direction
# to quantify which axis better separates the classes given their variance.

def within_class_variance(acts, idx_opp, idx_ded, layer=24):
    print(f"\n{'='*72}")
    print(f"  WITHIN-CLASS VARIANCE ANALYSIS (Layer {layer})")
    print(f"{'='*72}")

    X_opp = acts[idx_opp, layer, :]
    X_ded = acts[idx_ded, layer, :]

    var_opp = X_opp.var(axis=0).sum()
    var_ded = X_ded.var(axis=0).sum()
    print(f"\n  Total variance:")
    print(f"    opp-mod (n={len(idx_opp)}):   {var_opp:.2f}")
    print(f"    deduction (n={len(idx_ded)}): {var_ded:.2f}")
    print(f"    Ratio (ded/opp):               {var_ded/var_opp:.3f}")

    for name, X_class in [("opp-mod", X_opp), ("deduction", X_ded)]:
        pca = PCA(n_components=min(50, X_class.shape[0]-1))
        pca.fit(X_class)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        dim_80 = int(np.searchsorted(cumvar, 0.80)) + 1
        dim_50 = int(np.searchsorted(cumvar, 0.50)) + 1
        print(f"\n  {name} effective dimensionality:")
        print(f"    PCs for 50% variance: {dim_50}")
        print(f"    PCs for 80% variance: {dim_80}")
        print(f"    PCs for 90% variance: {dim_90}")
        print(f"    Top-1 PC explains:    {100*pca.explained_variance_ratio_[0]:.1f}%")

    u_opp = wvw_vectors["opponent_modeling"][layer]
    u_ded = wvw_vectors["deduction"][layer]
    dom_contrast = u_opp - u_ded
    dom_hat = dom_contrast / (np.linalg.norm(dom_contrast) + 1e-10)
    h_global = acts[:, layer, :].mean(axis=0)

    for name, X_class in [("opp-mod", X_opp), ("deduction", X_ded)]:
        X_centered = X_class - h_global
        proj = X_centered @ dom_hat
        var_along = proj.var()
        var_total = X_centered.var(axis=0).sum()
        frac = var_along / var_total
        print(f"\n  {name} variance along DoM contrast direction:")
        print(f"    Var along DoM:  {var_along:.2f}")
        print(f"    Total var:      {var_total:.2f}")
        print(f"    Fraction:       {100*frac:.3f}%")

    X_opp_c = X_opp - h_global
    X_ded_c = X_ded - h_global
    proj_opp = X_opp_c @ dom_hat
    proj_ded = X_ded_c @ dom_hat
    fisher_dom = (proj_opp.mean() - proj_ded.mean())**2 / (proj_opp.var() + proj_ded.var() + 1e-10)

    print(f"\n  Fisher's discriminant ratio along DoM contrast:")
    print(f"    (mean_opp - mean_ded)^2 / (var_opp + var_ded) = {fisher_dom:.4f}")
    print(f"    mean_opp={proj_opp.mean():+.2f}, mean_ded={proj_ded.mean():+.2f}")
    print(f"    std_opp={proj_opp.std():.2f}, std_ded={proj_ded.std():.2f}")

    X_all = np.concatenate([X_opp, X_ded], axis=0)
    y_all = np.concatenate([np.ones(len(idx_opp)), np.zeros(len(idx_ded))])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_all)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_s, y_all)
    w = clf.coef_[0] / scaler.scale_
    w_hat = w / (np.linalg.norm(w) + 1e-10)

    proj_opp_w = X_opp_c @ w_hat
    proj_ded_w = X_ded_c @ w_hat
    fisher_probe = (proj_opp_w.mean() - proj_ded_w.mean())**2 / (proj_opp_w.var() + proj_ded_w.var() + 1e-10)

    print(f"\n  Fisher's discriminant ratio along PROBE direction:")
    print(f"    {fisher_probe:.4f}")
    print(f"    mean_opp={proj_opp_w.mean():+.2f}, mean_ded={proj_ded_w.mean():+.2f}")
    print(f"    std_opp={proj_opp_w.std():.2f}, std_ded={proj_ded_w.std():.2f}")

    print(f"\n  Fisher ratio: probe/DoM = {fisher_probe/fisher_dom:.2f}x")
    if fisher_probe > 2 * fisher_dom:
        print(f"  → Probe direction has >2x the discriminant power of DoM.")
        print(f"    DoM captures mean separation but not the best separation given variance.")
    elif fisher_probe > fisher_dom:
        print(f"  → Probe is somewhat better than DoM. Modest improvement.")
    else:
        print(f"  → DoM is competitive with probe. Unusual given the low cosine.")


within_class_variance(acts, idx_opp, idx_ded, HEADLINE_LAYER)


# ===========================================================================
# Phase 2.5 Extended — Cell 11: 2D Scatter on SVD Components
# ===========================================================================
#
# Project all think-only segments onto the top 2 SVD components of the DoM
# category-mean matrix. Color by label to directly visualise whether
# categories form separable clusters. Also plots category-mean DoM vectors
# as stars for reference.

def plot_svd_scatter(acts, sids, label_masks, wvw_vectors, layer=24):
    print(f"\n{'='*72}")
    print(f"  2D SCATTER ON SVD COMPONENTS (Layer {layer})")
    print(f"{'='*72}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    svd_labels = [l for l in ALL_LABELS if l in wvw_vectors and label_masks[l].sum() >= 20]
    mat = np.stack([wvw_vectors[lab][layer] for lab in svd_labels], axis=0)
    U, S, Vt = np.linalg.svd(mat, full_matrices=False)

    h_global = acts[:, layer, :].mean(axis=0)
    X_centered = acts[:, layer, :] - h_global
    proj1 = X_centered @ Vt[0]
    proj2 = X_centered @ Vt[1]

    rng = np.random.RandomState(42)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    groups = [
        ("neither",  ~opp_mask & ~ded_mask, "lightgray", 5, 0.2),
        ("ded_only",  ded_only,              "steelblue", 15, 0.5),
        ("opp_only",  opp_only,              "crimson",   15, 0.5),
        ("BOTH",      both_mask,             "gold",      40, 0.9),
    ]
    for name, mask, color, size, alpha in groups:
        idx = np.where(mask)[0]
        if len(idx) > 1000:
            idx = rng.choice(idx, 1000, replace=False)
        ax.scatter(proj1[idx], proj2[idx], c=color, s=size, alpha=alpha,
                   label=f"{name} (n={mask.sum()})", edgecolors="none")
    ax.set_xlabel(f"SV1 ({100*(S[0]**2)/(S**2).sum():.1f}% var)")
    ax.set_ylabel(f"SV2 ({100*(S[1]**2)/(S**2).sum():.1f}% var)")
    ax.set_title(f"Segments projected onto DoM SVD components (L{layer})")
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.axvline(0, color="gray", ls="--", alpha=0.3)

    ax = axes[1]
    colors = {
        "opponent_modeling":        "crimson",
        "deduction":                "steelblue",
        "payoff_analysis":          "orange",
        "strategic_uncertainty":    "green",
        "equilibrium_identification": "purple",
        "initialization":           "gray",
        "none_other":               "black",
    }
    for lab, color in colors.items():
        if lab not in label_masks:
            continue
        mask = label_masks[lab]
        idx = np.where(mask)[0]
        if len(idx) > 500:
            idx = rng.choice(idx, 500, replace=False)
        ax.scatter(proj1[idx], proj2[idx], c=color, s=10, alpha=0.3,
                   label=f"{SHORT.get(lab, lab)} ({mask.sum()})", edgecolors="none")

    for lab in svd_labels:
        cm = wvw_vectors[lab][layer]
        p1 = np.dot(cm, Vt[0])
        p2 = np.dot(cm, Vt[1])
        color = colors.get(lab, "gray")
        ax.scatter(p1, p2, c=color, s=200, marker="*", edgecolors="black",
                   linewidths=1, zorder=10)
        ax.annotate(SHORT.get(lab, lab), (p1, p2), fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(f"SV1 ({100*(S[0]**2)/(S**2).sum():.1f}% var)")
    ax.set_ylabel(f"SV2 ({100*(S[1]**2)/(S**2).sum():.1f}% var)")
    ax.set_title(f"All labels + DoM means (stars) (L{layer})")
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.axvline(0, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    out_path = Path(OUTPUT_DIR) / f"svd_scatter_L{layer}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {out_path}")


plot_svd_scatter(acts, sids, label_masks, wvw_vectors, HEADLINE_LAYER)


# ===========================================================================
# Phase 2.5 Extended — Cell 12: Probe Direction Stability Across Layers
# ===========================================================================
#
# Do probes at different layers find the same discriminative direction?
# If yes, there is a single consistent discriminative axis through the model
# (even if it is not the DoM axis). Adjacent-layer cosine quantifies how
# quickly the probe direction drifts with depth.

def probe_direction_stability(probe_results):
    print(f"\n{'='*72}")
    print(f"  PROBE DIRECTION STABILITY ACROSS LAYERS")
    print(f"{'='*72}")

    layers_with_w = [(r["layer"], r["w_unscaled"]) for r in probe_results
                     if "w_unscaled" in r]

    if len(layers_with_w) < 5:
        print("  Not enough layers with saved weight vectors.")
        return

    ref_w = None
    for l, w in layers_with_w:
        if l == HEADLINE_LAYER:
            ref_w = w
            break

    if ref_w is None:
        print(f"  No probe weight for L{HEADLINE_LAYER}")
        return

    print(f"\n  Cosine of each layer's probe direction vs L{HEADLINE_LAYER} probe:")
    print(f"  {'Layer':>6}  {'cos(w_L, w_L24)':>16}  {'CV Acc':>8}")
    print(f"  {'-'*35}")

    cos_with_ref = []
    for r in probe_results:
        if "w_unscaled" not in r:
            continue
        c = cosine_sim(r["w_unscaled"], ref_w)
        cos_with_ref.append(c)
        marker = "  <--" if r["layer"] == HEADLINE_LAYER else ""
        print(f"  {r['layer']:>6}  {c:>+16.4f}  {r['cv_accuracy']:>8.4f}{marker}")

    cos_arr = np.array(cos_with_ref)
    print(f"\n  Summary:")
    print(f"    Mean cos(w_L, w_L24): {cos_arr.mean():.4f}")
    print(f"    Std:                  {cos_arr.std():.4f}")
    print(f"    Layers with cos > 0.5: {(cos_arr > 0.5).sum()}/48")
    print(f"    Layers with cos > 0.3: {(cos_arr > 0.3).sum()}/48")

    adj_cos = []
    for i in range(len(layers_with_w) - 1):
        l1, w1 = layers_with_w[i]
        l2, w2 = layers_with_w[i + 1]
        if l2 == l1 + 1:
            adj_cos.append(cosine_sim(w1, w2))

    if adj_cos:
        adj_arr = np.array(adj_cos)
        print(f"\n    Adjacent-layer cosine: mean={adj_arr.mean():.4f}, "
              f"min={adj_arr.min():.4f}, max={adj_arr.max():.4f}")
        if adj_arr.mean() > 0.8:
            print(f"    → Probe direction is STABLE across adjacent layers.")
            print(f"      A consistent discriminative axis exists through the model.")
        elif adj_arr.mean() > 0.5:
            print(f"    → Probe direction is MODERATELY stable.")
            print(f"      Similar but evolving direction across depth.")
        else:
            print(f"    → Probe direction CHANGES substantially between layers.")
            print(f"      No single consistent discriminative axis.")


probe_direction_stability(probe_results)


# ===========================================================================
# Save all results
# ===========================================================================

def save_all_results(probe_result, svd_result, cooccur_result, verdict_25,
                     probe_results, reg_results, probe_dir_result, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phase25 = {
        "probe": probe_result,
        "svd_depth": {
            "top1_pct": svd_result["top1_pct"],
            "top2_pct": svd_result["top2_pct"],
            "top3_pct": svd_result["top3_pct"],
            "labels":   svd_result["labels"],
        },
        "cooccurrence": {k: v for k, v in cooccur_result.items()
                         if isinstance(v, dict)},
        "verdict": verdict_25,
    }
    outfile = out / "phase25_results.json"
    with open(outfile, "w") as f:
        json.dump(phase25, f, indent=2, default=str)
    print(f"Saved: {outfile}")

    probe_clean = [{k: v for k, v in r.items() if k != "w_unscaled"}
                   for r in probe_results]
    extended = {
        "probe_all_layers":         probe_clean,
        "regularization_sweep":     reg_results,
        "probe_direction_analysis": probe_dir_result,
    }
    outfile = out / "phase25_extended_results.json"
    with open(outfile, "w") as f:
        json.dump(extended, f, indent=2, default=str)
    print(f"Saved: {outfile}")


save_all_results(
    probe_result, svd_result, cooccur_result, verdict_25,
    probe_results, reg_results, probe_dir_result, OUTPUT_DIR,
)

print("\nPhase 2 + Phase 2.5 complete.")
print("Next: Phase 3 (causal necessity via projection-out ablation).")
