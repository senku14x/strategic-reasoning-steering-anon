"""
Phase 0 annotation pipeline for the strategic-reasoning steering project.

Rewrite of the old mutually-exclusive / priority-ruled annotator. Changes:
  - Multi-label output (1-3 labels per segment, no priority hierarchy).
  - Per-segment API call with +/-1 context window (neighboring segments shown
    as context, clearly marked as non-labelable).
  - Explicit 'none_other' label for filler/transition/off-task segments.
  - Label definitions, canonical examples, and counter-examples carried as
    structured data; prompt is built from them so updates are cheap.
  - Answer-region segments labeled normally; region stored as a field so
    Phase 2 extraction can filter region at analysis time.
  - No silent fallbacks: parse failures, invalid labels, and moderation
    retries are all surfaced as explicit statuses / flags.

Examples for each label are TODO until Phase 0.5 gold sampling is done. Fill
them in LABEL_EXAMPLES before large-scale runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

try:
    from google.colab import userdata  # type: ignore
except ImportError:
    userdata = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("annotate_v2")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
if not logger.handlers:
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# Label set, definitions, and examples
# ---------------------------------------------------------------------------

ALLOWED_LABELS: List[str] = [
    "opponent_modeling",
    "iterated_reasoning",
    "equilibrium_identification",
    "payoff_analysis",
    "strategic_uncertainty",
    "cooperative_reasoning",
    "initialization",
    "deduction",
    "backtracking",
    "none_other",
]

# Definitions are the authoritative spec for each label.
# - 'definition': 1-3 sentences. States what the label IS and what it is NOT
#   with respect to nearest-neighbor labels.
# - 'positive_examples': 2-3 real segments that should receive this label.
#   Leave as [] until Phase 0.5 gold sampling surfaces real exemplars.
# - 'counter_examples': 1-2 segments that look like they might receive this
#   label but do not, with a brief reason.
#
# Rules we commit to in these definitions:
#   * Multi-label is normal. A segment that contains multiple cognitive
#     activities receives multiple labels.
#   * No priority hierarchy between "strategic" and "general" labels.
#   * 'backtracking' is orthogonal to content labels and co-occurs with them.
#   * 'none_other' is for segments that fit none of the content labels; it is
#     NOT a default for uncertainty. If the annotator is unsure between two
#     content labels, they should apply both rather than falling back to
#     none_other.

LabelSpec = Dict[str, object]

# Each example is a dict:
#   {
#     "text":   the segment text (short; excerpted if long)
#     "labels": labels that should be assigned to this segment
#     "reason": a one-sentence rationale explaining the labeling choice
#   }
# Positive examples show correct applications of THIS label (plus any other
# labels it co-occurs with). Counter-examples show segments where THIS label
# could look plausible but is wrong, with a reason explaining what is
# actually assigned instead.
#
# All examples below are drawn from hand-labeled gold segments (Phase 0.5).

LABEL_SPECS: Dict[str, LabelSpec] = {
    "opponent_modeling": {
        "definition": (
            "Reasoning about what another specific agent believes, knows, wants, "
            "will do, or is likely to do. The other agent is referenced explicitly "
            "(opponent, bidder, player, rival, them, he/she/they used referentially). "
            "NOT: reasoning about unknown variables that are not agents."
        ),
        "positive_examples": [
            {
                "text": "If I choose Low, they choose High to get $8 instead of $5. So, they choose High, and I get $10.",
                "labels": ["opponent_modeling", "payoff_analysis"],
                "reason": "Models the opponent's choice as a function of their payoff; includes the payoff numbers that drive the prediction.",
            },
            {
                "text": "But since I'm choosing after observing G, I can influence their prior choice.",
                "labels": ["opponent_modeling"],
                "reason": "Pure reasoning about how one's action affects another agent's choice; no payoff computation.",
            },
            {
                "text": "I think the key is that choosing Large is better for me, given that others choose Medium.",
                "labels": ["opponent_modeling", "payoff_analysis"],
                "reason": "Best response reasoning contingent on others' action; implicit payoff comparison drives 'better'.",
            },
        ],
        "counter_examples": [
            {
                "text": "But I'm not sure. Maybe I should consider that the signals are somewhat informative.",
                "labels": ["strategic_uncertainty", "backtracking"],
                "reason": "No referenced agent; this is uncertainty about signals, not modeling another agent.",
            },
        ],
    },
    "iterated_reasoning": {
        "definition": (
            "Higher-order belief reasoning, i.e. reasoning about what an agent "
            "thinks another agent thinks (k >= 2 levels of nested belief). "
            "Typical markers: 'they think I think', 'if they believe I will ...'. "
            "RARE label - apply only when explicit nested belief is present. "
            "NOT: first-order reasoning about an opponent's action (that is "
            "opponent_modeling). A segment can be both iterated_reasoning AND "
            "opponent_modeling."
        ),
        "positive_examples": [
            {
                "text": "If I decide to Free-ride, I need to consider whether the project will still succeed. If I Free-ride, I hope that at least 3 others will still contribute. But if others are also thinking the same way, maybe they'll also Free-ride, leading to fewer than 3 contributors and the project failing.",
                "labels": ["opponent_modeling", "iterated_reasoning", "strategic_uncertainty"],
                "reason": "Explicit nested belief: 'if others are also thinking the same way' reasons about others' beliefs about the situation, not just their actions.",
            },
        ],
        "counter_examples": [
            {
                "text": "If I choose Low, they choose High to get $8 instead of $5.",
                "labels": ["opponent_modeling", "payoff_analysis"],
                "reason": "First-order: reasons about what they do, not what they think about me. This is opponent_modeling only.",
            },
            {
                "text": "I think the key is that choosing Large is better for me, given that others choose Medium.",
                "labels": ["opponent_modeling", "payoff_analysis"],
                "reason": "Conditioned on others' action but no nested belief about others' beliefs. Not iterated.",
            },
        ],
    },
    "equilibrium_identification": {
        "definition": (
            "Identifying or invoking a game-theoretic solution concept by name "
            "or by structure: Nash equilibrium, dominant strategy, subgame-"
            "perfect equilibrium, mixed-strategy equilibrium, Bayesian Nash "
            "equilibrium, grim trigger, dominated strategies, etc. "
            "NOT: merely comparing two payoff numbers (that is payoff_analysis)."
        ),
        "positive_examples": [
            {
                "text": "Therefore, neither Medium nor High are dominant strategies for me.",
                "labels": ["equilibrium_identification"],
                "reason": "Explicitly reasoning about dominant strategies as a solution concept.",
            },
            {
                "text": "However, since none of the remaining rows are dominated, I can't eliminate any further. Therefore, the possible strategies I should consider are A, B, and D.",
                "labels": ["equilibrium_identification"],
                "reason": "Invokes iterated elimination of dominated strategies as a solution method.",
            },
            {
                "text": "If I choose Low and competitor chooses Medium, that's not an equilibrium because I would want to switch to Medium.",
                "labels": ["equilibrium_identification", "payoff_analysis"],
                "reason": "Equilibrium check via best-response deviation; also computes the payoff comparison that drives the switch.",
            },
        ],
        "counter_examples": [
            {
                "text": "premium_A = $2,400",
                "labels": ["payoff_analysis"],
                "reason": "Pure arithmetic; no solution concept invoked.",
            },
        ],
    },
    "payoff_analysis": {
        "definition": (
            "Computing, comparing, or reasoning over concrete numerical "
            "outcomes, utilities, costs, or expected values. Numbers or "
            "arithmetic are typically present, OR the segment references "
            "specific payoff values (e.g. '$6M', 'lose 15') to drive a "
            "strategic choice. "
            "NOT: pure symbolic algebra without payoff meaning (that is "
            "deduction), NOT: abstract solution concepts (that is "
            "equilibrium_identification)."
        ),
        "positive_examples": [
            {
                "text": "premium_A = $2,400",
                "labels": ["payoff_analysis"],
                "reason": "Computes a concrete dollar value; clean example of arithmetic over a payoff variable.",
            },
            {
                "text": "E(B) = 4",
                "labels": ["payoff_analysis"],
                "reason": "Expected value computation; single-label arithmetic.",
            },
            {
                "text": "So, if I Free-ride, and others also Free-ride, the project fails, and I lose 15.",
                "labels": ["opponent_modeling", "payoff_analysis"],
                "reason": "Concrete payoff number ('lose 15') drives the reasoning about an agent's choice; co-occurs with opponent_modeling.",
            },
        ],
        "counter_examples": [
            {
                "text": "3 > 6δ",
                "labels": ["deduction"],
                "reason": "Pure symbolic inequality; no payoff meaning attached to the variables. Deduction only.",
            },
            {
                "text": "Therefore, neither Medium nor High are dominant strategies for me.",
                "labels": ["equilibrium_identification"],
                "reason": "Strategic conclusion from solution concept, not numerical computation. No numbers present.",
            },
        ],
    },
    "strategic_uncertainty": {
        "definition": (
            "Reasoning about unknown information in a strategic setting: "
            "hidden types, private values, mixed strategies as a response to "
            "uncertainty, updating beliefs over distributions, posterior "
            "probabilities, bluffing considerations. The uncertainty is about "
            "something an agent does or knows. "
            "NOT: general epistemic hedging like 'I'm not sure'; that is "
            "none_other or deduction depending on context."
        ),
        "positive_examples": [
            {
                "text": "But without any binding commitment, I can't be sure. So, I need to decide based on what's more likely.",
                "labels": ["strategic_uncertainty"],
                "reason": "Reasoning about the strategic implications of not knowing whether commitment will hold; decision under uncertainty.",
            },
            {
                "text": "Given the high posterior probability that the opponent is TFT (98.43%), it's rational to expect that the opponent will follow the TFT strategy.",
                "labels": ["opponent_modeling", "strategic_uncertainty"],
                "reason": "Posterior belief update over opponent type is textbook strategic_uncertainty; also opponent_modeling because the belief is about the opponent.",
            },
            {
                "text": "11. Bluffing Consideration: Betting could be a bluff, but against an optimal opponent, this is riskier.",
                "labels": ["opponent_modeling", "strategic_uncertainty"],
                "reason": "Bluffing is a canonical strategic-uncertainty scenario (uncertainty about opponent's type/hand).",
            },
        ],
        "counter_examples": [
            {
                "text": "But I'm not entirely sure.",
                "labels": ["none_other"],
                "reason": "General hedging with no strategic content attached; filler, not strategic_uncertainty.",
            },
        ],
    },
    "cooperative_reasoning": {
        "definition": (
            "Analyzing coordination, joint optimization, fairness, trust, or "
            "enforcement of cooperative outcomes. Includes grim trigger, tit-"
            "for-tat sustainability analysis, coalition formation, fair-split "
            "reasoning. "
            "NOT: cooperation as a one-off strategic choice without analysis "
            "of the coordination structure. The word 'cooperate' or "
            "'contribute' appearing is not sufficient."
        ),
        "positive_examples": [
            {
                "text": "Grim trigger can sustain cooperation at δ = 0.8, but it may not be the best approach due to its harshness and lack of forgiveness.",
                "labels": ["equilibrium_identification", "cooperative_reasoning"],
                "reason": "Analyzes a specific cooperation-enforcing mechanism and its tradeoffs; also invokes a solution concept.",
            },
            {
                "text": "Therefore, I can offer C $10, keep $190, and that's a stable coalition.",
                "labels": ["equilibrium_identification", "payoff_analysis", "cooperative_reasoning"],
                "reason": "Coalition stability analysis - cooperative structure plus payoff numbers plus equilibrium concept.",
            },
            {
                "text": "Assume that both players agree to play (High, High) every period, getting 6 each. If one deviates to Low, they get 9 in that period, but then the other player will punish them.",
                "labels": ["payoff_analysis", "cooperative_reasoning"],
                "reason": "Analyzes how deviation payoffs and punishment sustain cooperation; structural cooperative reasoning.",
            },
        ],
        "counter_examples": [
            {
                "text": "So, perhaps the best strategy is to Contribute, to avoid losing 15.",
                "labels": ["payoff_analysis", "deduction"],
                "reason": "Just picks 'Contribute' out of loss-avoidance, with no analysis of the cooperative structure. Action name is not the label.",
            },
        ],
    },
    "initialization": {
        "definition": (
            "Actively parsing or restating the problem structure: listing "
            "players, enumerating actions, restating payoffs or rules as given, "
            "setting up variable definitions to be used later. "
            "NOT: off-task filler, short section headers, or narrative "
            "connective tissue (those are none_other)."
        ),
        "positive_examples": [
            {
                "text": "The payoffs are given as:",
                "labels": ["initialization"],
                "reason": "Restating the given problem structure (payoffs).",
            },
            {
                "text": "First, I need to understand what discounting means here. The problem says I discount at 5% per round, and my opponent discounts at 15%.",
                "labels": ["payoff_analysis", "initialization"],
                "reason": "Parses and restates the given discount rates from the problem; also numerical. Multi-label.",
            },
            {
                "text": "First, let me recall the setup. Both my rival and I have been using grim trigger strategies for the past 20 rounds.",
                "labels": ["cooperative_reasoning", "initialization"],
                "reason": "Restating the problem setup explicitly ('let me recall the setup').",
            },
        ],
        "counter_examples": [
            {
                "text": "For Low-quality:",
                "labels": ["none_other"],
                "reason": "Short section header, not active problem parsing. This is filler / none_other.",
            },
            {
                "text": "Alright, so I'm trying to figure out what number I should pick in this Keynesian beauty contest variant to minimize my expected loss. Let me break it down step by step.",
                "labels": ["none_other"],
                "reason": "Narrative opener expressing intent to reason, not actually restating problem structure. None_other.",
            },
        ],
    },
    "deduction": {
        "definition": (
            "Step-by-step logical derivation, procedural computation, or "
            "'if X then Y' inference chains that do not themselves fit a more "
            "specific content label. Most common in pure-math / non-strategic "
            "reasoning (control tasks) and in bare algebraic manipulation "
            "without payoff semantics. "
            "Do NOT use deduction as a fallback for uncertainty; if no content "
            "label fits, use none_other."
        ),
        "positive_examples": [
            {
                "text": "3 > 6δ",
                "labels": ["deduction"],
                "reason": "Pure symbolic inequality step; no payoff or strategic content.",
            },
            {
                "text": "Solve for x: Add 15 to both sides to isolate x: x = 24",
                "labels": ["deduction"],
                "reason": "Algebraic manipulation; classic deduction.",
            },
            {
                "text": "Determine the Remaining Distance: Total distance = 450 km. Distance covered = 135 km. Remaining distance = 450 - 135 = 315 km.",
                "labels": ["deduction"],
                "reason": "Non-strategic control-task arithmetic; no payoff meaning. Clean deduction.",
            },
        ],
        "counter_examples": [
            {
                "text": "premium_A = $2,400",
                "labels": ["payoff_analysis"],
                "reason": "Arithmetic over a strategic quantity (a premium). Payoff_analysis, not deduction.",
            },
            {
                "text": "If I choose Low, they choose High.",
                "labels": ["opponent_modeling"],
                "reason": "'If X then Y' inference but about agent behavior. Opponent_modeling, not deduction.",
            },
        ],
    },
    "backtracking": {
        "definition": (
            "Correcting an earlier error, revising a prior conclusion, or "
            "explicitly retracting a previous step. Typical markers: 'wait', "
            "'actually', 'but earlier I thought', 'no, that's not right'. "
            "CROSS-CUTTING: a segment that retracts a payoff computation "
            "receives both backtracking AND payoff_analysis. "
            "NOT: mere continuation of a calculation; NOT: uncertainty "
            "expressions that don't revise a prior step."
        ),
        "positive_examples": [
            {
                "text": "Wait, but the cost per bid is $1, and the price increase is $0.05. So, each bid is a net loss of $0.95. That doesn't make sense because I'm losing money on each bid.",
                "labels": ["payoff_analysis", "backtracking"],
                "reason": "'Wait' signals revision; re-examines the payoff computation from a different angle.",
            },
            {
                "text": "Wait, no, that's not the right approach. Maybe I should think of it as the sum of all pair values minus the individual values.",
                "labels": ["payoff_analysis", "backtracking"],
                "reason": "Explicit retraction ('that's not the right approach') of a prior payoff analysis method.",
            },
            {
                "text": "But earlier, I thought that if I choose Low, competitor can choose Medium and get $6M, which is better for them, leading to me getting -$1M.",
                "labels": ["opponent_modeling", "payoff_analysis", "backtracking"],
                "reason": "'But earlier I thought' explicitly revisits a prior opponent-modeling step with payoffs attached.",
            },
        ],
        "counter_examples": [
            {
                "text": "But I'm not entirely sure.",
                "labels": ["none_other"],
                "reason": "Hedging, not retraction of a specific prior step. Filler.",
            },
            {
                "text": "But since I'm choosing after observing G, I can influence their prior choice.",
                "labels": ["opponent_modeling"],
                "reason": "'But' introduces a new observation, not a correction of earlier reasoning. Not backtracking.",
            },
        ],
    },
    "none_other": {
        "definition": (
            "Filler, transitions, off-task comments, short section headers "
            "(e.g. 'For Low-quality:', 'If I Call:'), meta-remarks about the "
            "reasoning process ('I'm getting stuck here', 'Let me verify'), "
            "narrative openers ('Alright, let me break this down'), or content "
            "that genuinely fits no content label. "
            "Use this when no content label applies; do NOT use it as a tie-"
            "breaker when multiple content labels could apply (in that case, "
            "apply all that fit)."
        ),
        "positive_examples": [
            {
                "text": "For Low-quality:",
                "labels": ["none_other"],
                "reason": "Short section header; no reasoning content by itself.",
            },
            {
                "text": "Let me verify:",
                "labels": ["none_other"],
                "reason": "Meta-remark about the reasoning process, not reasoning itself.",
            },
            {
                "text": "But I'm not entirely sure.",
                "labels": ["none_other"],
                "reason": "General hedging with no strategic content; filler.",
            },
            {
                "text": "I'm getting stuck here.",
                "labels": ["none_other"],
                "reason": "Meta-commentary on the reasoning process.",
            },
        ],
        "counter_examples": [
            {
                "text": "The payoffs are given as:",
                "labels": ["initialization"],
                "reason": "Actively parsing the problem (introducing a payoff restatement); not filler. Initialization.",
            },
            {
                "text": "Therefore, I should choose Low.",
                "labels": ["payoff_analysis"],
                "reason": "A decision conclusion based on prior payoff reasoning; has strategic content. Not none_other.",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

PROMPT_VERSION = "phase0_multilabel_v1"

_HEADER = (
    "You are an annotator for academic research on strategic reasoning in "
    "language models. You will label a single TARGET segment from a chain-"
    "of-thought, with its immediate neighbors shown as CONTEXT (do NOT label "
    "context segments).\n"
    "\n"
    "Assign EVERY label that applies to the TARGET segment, as a list.\n"
    "\n"
    "Multi-label is the norm. Most reasoning segments contain two or more "
    "cognitive activities simultaneously (e.g. computing payoffs AND modeling "
    "an opponent's likely response). Assigning only one label should be the "
    "exception, not the default. Before finalizing, check whether a second or "
    "third label also fits.\n"
    "\n"
    "Assign between 1 and 3 labels. If no content label fits, assign "
    "'none_other' (not 'deduction').\n"
)

_RULES = (
    "\nRules:\n"
    "- Do NOT quote, reproduce, or paraphrase the segment text in your output.\n"
    "- Do NOT include any label that is not in the allowed list.\n"
    "- Labels are symmetric: there is no priority between 'strategic' and "
    "'general' labels. Apply all that fit.\n"
    "- 'backtracking' is cross-cutting: a segment that revises a payoff "
    "computation receives both 'backtracking' and 'payoff_analysis'.\n"
    "- 'none_other' is NOT a fallback for uncertainty between two content "
    "labels. If unsure, apply both.\n"
)


# When running in chain mode (many segments in one API call), we need to
# explicitly instruct the model against the specific contamination failure
# modes whole-chain batching invites. These instructions are REPLACED INTO
# the header in chain mode (they don't apply in per-segment mode, where
# each segment is already judged independently by construction).
CHAIN_MODE_ADDENDUM = (
    "\n\n"
    "CHAIN MODE: You will be given ALL segments of one reasoning chain in "
    "this call. Label EVERY segment. Return a list with one entry per "
    "segment index.\n"
    "\n"
    "CRITICAL independence requirement:\n"
    "- Judge each segment INDEPENDENTLY. Do NOT let your label for one "
    "segment influence your label for any other segment.\n"
    "- Do NOT seek consistency across segments. Adjacent segments frequently "
    "carry different labels. A chain might have payoff_analysis in segment "
    "2, opponent_modeling alone in segment 3, and payoff_analysis + "
    "backtracking in segment 4. Do NOT smooth these assignments.\n"
    "- Do NOT reuse labels just because they appeared nearby. The fact that "
    "segment 3 was labeled 'opponent_modeling' is NOT evidence about "
    "segment 4.\n"
    "- Do NOT apply a 'chain-wide theme' label to every segment. Each "
    "segment must earn its label from its OWN content.\n"
    "- A segment near the END of the chain deserves exactly as much care "
    "as a segment near the beginning. Do not let labeling quality degrade "
    "as you move through the list.\n"
)


def _format_label_spec(name: str, spec: LabelSpec) -> str:
    lines = [f"### {name}", str(spec["definition"])]
    pos = spec["positive_examples"]  # type: ignore[index]
    neg = spec["counter_examples"]   # type: ignore[index]

    def _render_example(ex) -> str:
        # Legacy string form (fallback) or new dict form.
        if isinstance(ex, str):
            return f"  - {ex}"
        text = str(ex.get("text", "")).replace("\n", " ").strip()
        labels = ex.get("labels", [])
        reason = str(ex.get("reason", "")).strip()
        labs_str = ", ".join(labels) if labels else ""
        out = f"  - {text!r}  ->  labels=[{labs_str}]"
        if reason:
            out += f"\n      ({reason})"
        return out

    if pos:
        lines.append("Positive examples (correct applications of this label):")
        for ex in pos:  # type: ignore[union-attr]
            lines.append(_render_example(ex))
    if neg:
        lines.append("Counter-examples (look similar but do NOT get this label):")
        for ex in neg:  # type: ignore[union-attr]
            lines.append(_render_example(ex))
    return "\n".join(lines)


def build_system_prompt(chain_mode: bool = False) -> str:
    """
    Build the system prompt. Set chain_mode=True to append the anti-
    contamination addendum required for whole-chain batching.
    """
    parts = [_HEADER]
    if chain_mode:
        parts.append(CHAIN_MODE_ADDENDUM)
    parts.append("\nAllowed labels (use EXACTLY these strings):\n")
    parts.extend(f"  - {name}" for name in ALLOWED_LABELS)
    parts.append("\n\nLabel definitions:\n")
    for name in ALLOWED_LABELS:
        parts.append(_format_label_spec(name, LABEL_SPECS[name]))
        parts.append("")
    parts.append(_RULES)
    return "\n".join(parts)


SYSTEM_PROMPT = build_system_prompt(chain_mode=False)
SYSTEM_PROMPT_CHAIN = build_system_prompt(chain_mode=True)


# ---------------------------------------------------------------------------
# Pydantic schema for multi-label output
# ---------------------------------------------------------------------------

class SegmentLabels(BaseModel):
    """Multi-label response for a single target segment."""
    labels: List[str] = Field(..., min_length=1, max_length=3)

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for lab in v:
            if lab not in ALLOWED_LABELS:
                raise ValueError(f"invalid label: {lab!r}")
            if lab in seen:
                continue  # drop exact duplicates silently (model sometimes repeats)
            seen.add(lab)
            out.append(lab)
        return out


class OneChainSegmentLabels(BaseModel):
    """Labels for a single segment within a whole-chain response."""
    i: int = Field(..., ge=0)
    labels: List[str] = Field(..., min_length=1, max_length=3)

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for lab in v:
            if lab not in ALLOWED_LABELS:
                raise ValueError(f"invalid label: {lab!r}")
            if lab in seen:
                continue
            seen.add(lab)
            out.append(lab)
        return out


class ChainLabels(BaseModel):
    """Multi-label response covering every segment of a reasoning chain."""
    segments: List[OneChainSegmentLabels] = Field(..., min_length=1)

    @field_validator("segments")
    @classmethod
    def _validate_unique_indices(cls, v: List[OneChainSegmentLabels]) -> List[OneChainSegmentLabels]:
        seen_i: set = set()
        for s in v:
            if s.i in seen_i:
                raise ValueError(f"duplicate segment index: {s.i}")
            seen_i.add(s.i)
        return v


# ---------------------------------------------------------------------------
# Segmentation (unchanged from old pipeline, kept for parity)
# ---------------------------------------------------------------------------

# Sentinel emitted by the generation script when generation hits max_tokens
# inside <think> and there is no real post-think answer. Treat as empty.
SENTINEL_NO_ANSWER = "[No separate answer - see thinking]"


def parse_chain(record: Dict) -> Tuple[str, str, bool, str]:
    """
    Extract reasoning/answer text from a chain record.

    Returns (thinking_text, answer_text, was_truncated, source).

    Behavior:
      - Prefers pre-parsed `thinking` and `answer` fields if present.
      - Drops the `[No separate answer - see thinking]` sentinel.
      - Detects truncation: a chain is `was_truncated` if `<think>` was emitted
        but `</think>` was not (generation hit max_tokens mid-reasoning). When
        truncated, all content belongs to the thinking region; answer is empty.
      - If pre-parsed fields are absent, falls back to parsing `full_output`.
        The fallback ALSO routes truncated content to thinking, not answer
        (this is the bug-fix vs. the old `split_think_answer`, which dumped
        truncated thinking into the answer region).

    `source` is one of: "fields" | "full_output" | "none".
    """
    fo = (record.get("full_output") or "")
    has_open = "<think>" in fo
    has_close = "</think>" in fo
    was_truncated = has_open and not has_close

    # Prefer pre-parsed fields.
    pre_think = (record.get("thinking") or "").strip()
    pre_answer = (record.get("answer") or "").strip()
    if pre_answer == SENTINEL_NO_ANSWER:
        pre_answer = ""
    if was_truncated:
        # Generation hit max_tokens inside <think>; answer is meaningless.
        pre_answer = ""

    if pre_think or pre_answer:
        return pre_think, pre_answer, was_truncated, "fields"

    # Fallback: parse from full_output.
    if not fo.strip():
        return "", "", False, "none"

    a = fo.find("<think>")
    b = fo.find("</think>")
    if a != -1 and b != -1 and b > a:
        think = fo[a + len("<think>"):b].strip()
        ans = fo[b + len("</think>"):].strip()
        return think, ans, False, "full_output"
    if a != -1:
        # truncated: everything after <think> is reasoning, no answer
        think = fo[a + len("<think>"):].strip()
        return think, "", True, "full_output"
    # No <think> tag at all: unusual; treat the whole thing as answer.
    return "", fo.strip(), False, "full_output"


def paragraph_split(text: str) -> List[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def hard_wrap_chunks(text: str, max_chars: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start + int(0.6 * max_chars):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]


def make_segments(
    thinking_text: str,
    answer_text: str,
    max_seg_chars: int = 1800,
) -> List[Dict]:
    """
    Build the per-segment list from already-parsed thinking and answer text.

    Each segment carries: i (global index), region ("thinking"|"answer"), text.
    Paragraph-level split (matches old pipeline). Hard-wrap fallback only
    triggers for paragraphs above max_seg_chars (rare on this dataset).
    """
    segments: List[Dict] = []
    idx = 0

    def add_region(region: str, region_text: str) -> None:
        nonlocal idx
        if not region_text:
            return
        for p in paragraph_split(region_text):
            if len(p) <= max_seg_chars:
                segments.append({"i": idx, "region": region, "text": p})
                idx += 1
            else:
                for c in hard_wrap_chunks(p, max_seg_chars):
                    segments.append({"i": idx, "region": region, "text": c})
                    idx += 1

    add_region("thinking", thinking_text)
    add_region("answer", answer_text)
    return segments


# ---------------------------------------------------------------------------
# Moderation redaction (logged, not silent)
# ---------------------------------------------------------------------------

_RISK_TERMS = [
    r"\bkill\b", r"\bmurder\b", r"\bshoot\b", r"\bstab\b", r"\bbomb\b",
    r"\bterror\b", r"\bsuicide\b", r"\bself[- ]harm\b", r"\boverdose\b",
]
_risk_re = re.compile("|".join(_RISK_TERMS), flags=re.IGNORECASE)


def redact_text(s: str) -> str:
    return _risk_re.sub("[REDACTED]", s)


# ---------------------------------------------------------------------------
# Context-window construction
# ---------------------------------------------------------------------------

def build_user_prompt(
    target: Dict,
    prev_seg: Optional[Dict],
    next_seg: Optional[Dict],
    task_id: str,
) -> str:
    """Render the target segment with +/-1 context neighbors."""
    parts: List[str] = [f"Task ID: {task_id}\n"]
    if prev_seg is not None:
        parts.append(
            f"[CONTEXT - previous segment, region={prev_seg['region']}, "
            f"do NOT label]\n{prev_seg['text']}\n"
        )
    parts.append(
        f"[TARGET - region={target['region']}, LABEL THIS]\n{target['text']}\n"
    )
    if next_seg is not None:
        parts.append(
            f"[CONTEXT - next segment, region={next_seg['region']}, "
            f"do NOT label]\n{next_seg['text']}\n"
        )
    parts.append(
        "\nReturn labels for the TARGET segment only, as a list of 1-3 "
        "labels from the allowed set."
    )
    return "\n".join(parts)


def build_chain_user_prompt(task_id: str, chain_segments: List[Dict]) -> str:
    """
    Render all segments of a chain for whole-chain annotation.

    Each segment is numbered; the model returns a list of (i, labels) pairs,
    one per segment index.
    """
    lines: List[str] = [
        f"Task ID: {task_id}",
        f"N_SEGMENTS = {len(chain_segments)}",
        "",
        "Segments:",
    ]
    for s in chain_segments:
        # Each segment gets an explicit integer index matching its `i` field.
        # Use a clear delimiter that won't collide with segment text.
        header = f"--- segment i={s['i']}  region={s['region']} ---"
        lines.append(header)
        lines.append(s["text"])
        lines.append("")  # blank line between segments
    lines.append(
        "Return labels for EVERY segment, as a list with one entry per "
        "segment index i. Each entry carries 1-3 labels from the allowed set. "
        "Judge each segment INDEPENDENTLY, without smoothing across segments."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-segment API call
# ---------------------------------------------------------------------------

async def annotate_one_segment(
    client: AsyncOpenAI,
    target: Dict,
    prev_seg: Optional[Dict],
    next_seg: Optional[Dict],
    task_id: str,
    model: str = "gpt-5.2",
    max_retries: int = 3,
) -> Dict:
    """
    Annotate a single target segment. Returns a dict with:
      - i: segment index
      - labels: list[str] on success, [] on failure
      - status: "ok" | "parse_failed" | "api_failed" | "redacted_ok"
      - error: Optional[str]
      - redacted: bool (True if moderation redaction was applied)
    """
    user_prompt = build_user_prompt(target, prev_seg, next_seg, task_id)
    cur_prompt = user_prompt
    redacted = False
    last_err: Optional[str] = None

    for attempt in range(max_retries):
        try:
            resp = await client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": cur_prompt},
                ],
                text_format=SegmentLabels,
                reasoning={"effort": "none"},
                temperature=0.0,
            )
            parsed: SegmentLabels = resp.output_parsed
            return {
                "i": target["i"],
                "labels": parsed.labels,
                "status": "redacted_ok" if redacted else "ok",
                "error": None,
                "redacted": redacted,
            }

        except Exception as e:  # noqa: BLE001 - surfaced explicitly
            msg = str(e)
            last_err = msg

            # Moderation failure: redact once, log task_id, retry.
            if ("invalid_prompt" in msg or "moderation" in msg.lower()) and not redacted:
                logger.warning(
                    "moderation_redaction task=%s seg=%d attempt=%d",
                    task_id, target["i"], attempt,
                )
                cur_prompt = redact_text(cur_prompt)
                redacted = True
                continue

            # Pydantic validation error -> this is not retriable with the same
            # prompt; the model returned an invalid label. Surface explicitly.
            if "validation error" in msg.lower() or "ValidationError" in msg:
                return {
                    "i": target["i"],
                    "labels": [],
                    "status": "parse_failed",
                    "error": msg,
                    "redacted": redacted,
                }

            # Transient errors: exponential backoff.
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue

    return {
        "i": target["i"],
        "labels": [],
        "status": "api_failed",
        "error": last_err or "unknown_error",
        "redacted": redacted,
    }


# ---------------------------------------------------------------------------
# Whole-chain API call (chain mode)
# ---------------------------------------------------------------------------

async def annotate_one_chain(
    client: AsyncOpenAI,
    task_id: str,
    chain_segments: List[Dict],
    model: str = "gpt-5.2",
    max_retries: int = 3,
) -> Dict:
    """
    Annotate every segment of a chain in one API call.

    `chain_segments` must be a list of dicts each having keys i, region, text,
    ordered by i ascending.

    Returns:
      {
        "task_id": ...,
        "status": "ok" | "redacted_ok" | "parse_failed" | "api_failed" |
                  "incomplete_coverage",
        "error":  Optional[str],
        "redacted": bool,
        # Per-segment labels, keyed by segment index i. Missing indices
        # (model failed to label them) are absent from this dict.
        "segment_labels": {i: [labels...], ...},
      }
    """
    user_prompt = build_chain_user_prompt(task_id, chain_segments)
    cur_prompt = user_prompt
    redacted = False
    last_err: Optional[str] = None
    expected_indices = {s["i"] for s in chain_segments}

    for attempt in range(max_retries):
        try:
            resp = await client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT_CHAIN},
                    {"role": "user", "content": cur_prompt},
                ],
                text_format=ChainLabels,
                reasoning={"effort": "none"},
                temperature=0.0,
            )
            parsed: ChainLabels = resp.output_parsed
            by_i: Dict[int, List[str]] = {s.i: s.labels for s in parsed.segments}

            # Validate coverage: every segment index we sent must appear.
            missing = expected_indices - set(by_i.keys())
            extras = set(by_i.keys()) - expected_indices
            if missing:
                # Partial coverage is treated as a hard failure so Phase 2
                # analysis doesn't silently operate on incomplete data.
                return {
                    "task_id": task_id,
                    "status": "incomplete_coverage",
                    "error": f"missing indices: {sorted(missing)[:10]}... "
                             f"({len(missing)} total)",
                    "redacted": redacted,
                    "segment_labels": by_i,
                }
            if extras:
                # Model returned labels for indices we didn't send.
                # Drop those; don't fail the whole chain for it.
                for k in extras:
                    by_i.pop(k, None)

            return {
                "task_id": task_id,
                "status": "redacted_ok" if redacted else "ok",
                "error": None,
                "redacted": redacted,
                "segment_labels": by_i,
            }

        except Exception as e:  # noqa: BLE001
            msg = str(e)
            last_err = msg

            if ("invalid_prompt" in msg or "moderation" in msg.lower()) and not redacted:
                logger.warning(
                    "chain_moderation_redaction task=%s attempt=%d",
                    task_id, attempt,
                )
                cur_prompt = redact_text(cur_prompt)
                redacted = True
                continue

            if "validation error" in msg.lower() or "ValidationError" in msg:
                return {
                    "task_id": task_id,
                    "status": "parse_failed",
                    "error": msg,
                    "redacted": redacted,
                    "segment_labels": {},
                }

            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue

    return {
        "task_id": task_id,
        "status": "api_failed",
        "error": last_err or "unknown_error",
        "redacted": redacted,
        "segment_labels": {},
    }


# ---------------------------------------------------------------------------
# Per-record orchestration
# ---------------------------------------------------------------------------

async def annotate_one_record(
    client: AsyncOpenAI,
    record: Dict,
    max_seg_chars: int,
    segment_concurrency: int,
    model: str,
) -> Dict:
    tid = record["task_id"]
    thinking_text, answer_text, was_truncated, source = parse_chain(record)

    if not thinking_text and not answer_text:
        return {
            **record,
            "annotated_field": source,
            "chain_truncated": was_truncated,
            "annotation_status": "no_content",
            "segments": [],
            "segment_labels": [],
        }

    segments = make_segments(thinking_text, answer_text, max_seg_chars=max_seg_chars)
    n = len(segments)
    if n == 0:
        return {
            **record,
            "annotated_field": source,
            "chain_truncated": was_truncated,
            "annotation_status": "no_content",
            "segments": [],
            "segment_labels": [],
        }

    sem = asyncio.Semaphore(segment_concurrency)
    results: List[Optional[Dict]] = [None] * n

    async def run_one(i: int) -> None:
        target = segments[i]
        prev_seg = segments[i - 1] if i > 0 else None
        next_seg = segments[i + 1] if i < n - 1 else None
        async with sem:
            results[i] = await annotate_one_segment(
                client=client,
                target=target,
                prev_seg=prev_seg,
                next_seg=next_seg,
                task_id=tid,
                model=model,
            )

    await asyncio.gather(*(run_one(i) for i in range(n)))

    # Compute record-level status.
    statuses = [r["status"] for r in results if r is not None]
    n_ok = sum(1 for s in statuses if s in ("ok", "redacted_ok"))
    n_parse_failed = sum(1 for s in statuses if s == "parse_failed")
    n_api_failed = sum(1 for s in statuses if s == "api_failed")

    if n_ok == n:
        record_status = "success"
    elif n_ok == 0:
        record_status = "failed"
    else:
        record_status = "partial"

    return {
        **record,
        "annotated_field": source,
        "chain_truncated": was_truncated,
        "annotation_status": record_status,
        "segments": segments,
        "segment_labels": results,
        "segment_stats": {
            "n": n,
            "ok": n_ok,
            "parse_failed": n_parse_failed,
            "api_failed": n_api_failed,
            "redacted": sum(1 for r in results if r and r.get("redacted")),
        },
    }


# ---------------------------------------------------------------------------
# File-level orchestration with resume + periodic save
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)  # atomic replace; no half-written outputs on crash


def save_progress(
    output_file: Path,
    data: Dict,
    results: List[Dict],
    status: str,
) -> None:
    n_truncated = sum(1 for r in results if r.get("chain_truncated"))
    n_total_segments = sum(len(r.get("segments", [])) for r in results)
    out = {
        **data,
        "results": results,
        "annotation_metadata": {
            "total_tasks": len(data["results"]),
            "completed_so_far": sum(
                1 for r in results if r.get("annotation_status") in ("success", "partial")
            ),
            "model": "gpt-5.2",
            "api": "responses.parse",
            "method": "per_segment_context_window_multilabel",
            "prompt_version": PROMPT_VERSION,
            "labels": ALLOWED_LABELS,
            "n_truncated_chains": n_truncated,
            "n_total_segments": n_total_segments,
            "status": status,
        },
    }
    save_json(output_file, out)


async def annotate_file(
    client: AsyncOpenAI,
    input_file: Path,
    output_file: Path,
    resume: bool,
    save_every: int,
    task_concurrency: int,
    segment_concurrency: int,
    max_seg_chars: int,
    model: str,
) -> Dict:
    data = load_json(input_file)
    total = len(data["results"])

    logger.info("annotating file=%s tasks=%d", input_file.name, total)

    existing: Dict[str, Dict] = {}
    if resume and output_file.exists():
        try:
            prev = load_json(output_file)
            for r in prev.get("results", []):
                # Only treat fully-successful records as complete. Partial
                # records get re-run to avoid persistent gaps.
                if r.get("annotation_status") == "success" and r.get("segment_labels"):
                    existing[r["task_id"]] = r
            logger.info("resume: found %d completed tasks", len(existing))
        except Exception as e:
            logger.warning("resume failed, starting fresh: %s", e)

    idx_map = {r["task_id"]: i for i, r in enumerate(data["results"])}
    out_results: List[Optional[Dict]] = [None] * total

    skipped = 0
    for r in data["results"]:
        tid = r["task_id"]
        if tid in existing:
            out_results[idx_map[tid]] = existing[tid]
            skipped += 1

    task_sem = asyncio.Semaphore(task_concurrency)
    stats = {"total": total, "annotated": 0, "skipped": skipped, "failed": 0, "partial": 0}

    async def run_one(r: Dict) -> None:
        tid = r["task_id"]
        if tid in existing:
            return
        async with task_sem:
            rec = await annotate_one_record(
                client=client,
                record=r,
                max_seg_chars=max_seg_chars,
                segment_concurrency=segment_concurrency,
                model=model,
            )
        out_results[idx_map[tid]] = rec
        st = rec["annotation_status"]
        if st == "success":
            stats["annotated"] += 1
        elif st == "partial":
            stats["partial"] += 1
        else:
            stats["failed"] += 1

    todo = [r for r in data["results"] if r["task_id"] not in existing]
    tasks = [asyncio.create_task(run_one(r)) for r in todo]

    pbar = tqdm(total=len(tasks), desc=f"Annotating {input_file.name}")
    completed = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        completed += 1
        pbar.update(1)
        if completed % save_every == 0:
            tmp = [
                x if x is not None else {**orig, "annotation_status": "pending"}
                for x, orig in zip(out_results, data["results"])
            ]
            save_progress(output_file, data, tmp, status="in_progress")
    pbar.close()

    final = [
        x if x is not None else {**orig, "annotation_status": "pending"}
        for x, orig in zip(out_results, data["results"])
    ]
    save_progress(output_file, data, final, status="complete")

    n_truncated = sum(1 for r in final if r.get("chain_truncated"))
    n_total_segments = sum(
        len(r.get("segments", [])) for r in final
    )

    logger.info(
        "done file=%s annotated=%d partial=%d failed=%d skipped=%d "
        "truncated_chains=%d total_segments=%d",
        input_file.name,
        stats["annotated"], stats["partial"], stats["failed"], stats["skipped"],
        n_truncated, n_total_segments,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_api_key(cli_arg: Optional[str]) -> str:
    if cli_arg:
        return cli_arg
    if userdata is not None:
        try:
            k = userdata.get("OPENAI_API_KEY")
            if k:
                return k
        except Exception:
            pass
    import os
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k
    raise ValueError(
        "No API key. Pass --api_key, set Colab secret OPENAI_API_KEY, or "
        "export OPENAI_API_KEY."
    )


async def amain() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 multi-label annotation (per-segment, +/-1 context)"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["r1-qwen-14b", "qwen-14b", "r1-llama-8b", "llama-8b", "all"],
        default=["all"],
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save progress every N completed tasks.")
    parser.add_argument("--task_concurrency", type=int, default=3,
                        help="Concurrent tasks (chains) in flight.")
    parser.add_argument("--segment_concurrency", type=int, default=8,
                        help="Concurrent segment annotations within a task.")
    parser.add_argument("--max_seg_chars", type=int, default=1200)
    parser.add_argument("--gpt_model", type=str, default="gpt-5.2")
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    client = AsyncOpenAI(api_key=api_key)

    file_map = {
        "r1-qwen-14b": ("r1_qwen14b_chains.json", "r1_qwen14b_annotated_v2.json"),
        "qwen-14b": ("qwen14b_base_chains.json", "qwen14b_base_annotated_v2.json"),
        "r1-llama-8b": ("r1_llama8b_chains.json", "r1_llama8b_annotated_v2.json"),
        "llama-8b": ("llama8b_base_chains.json", "llama8b_base_annotated_v2.json"),
    }

    models = list(file_map.keys()) if "all" in args.models else args.models

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("run: models=%s resume=%s model=%s", models, not args.no_resume, args.gpt_model)
    logger.info("concurrency: tasks=%d segments=%d", args.task_concurrency, args.segment_concurrency)

    for m in models:
        inp, outp = file_map[m]
        in_path = in_dir / inp
        out_path = out_dir / outp
        if not in_path.exists():
            logger.warning("skip %s: missing %s", m, in_path)
            continue
        await annotate_file(
            client=client,
            input_file=in_path,
            output_file=out_path,
            resume=not args.no_resume,
            save_every=args.save_every,
            task_concurrency=args.task_concurrency,
            segment_concurrency=args.segment_concurrency,
            max_seg_chars=args.max_seg_chars,
            model=args.gpt_model,
        )


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
