"""
Whole-chain annotation runner.

Counterpart to annotate_segments.py but sends ONE API call per chain instead
of one call per segment. Dramatically cheaper (~10x fewer calls × ~5x shorter
amortized prompt = ~50x cheaper in input tokens) at the cost of
cross-segment contamination risk, which the prompt mitigates via the
CHAIN_MODE_ADDENDUM instructions.

Output schema (per segment, in the same JSONL format as annotate_segments.py
so evaluate_gold.py can read it unchanged):
  {
    "segment_id": "matrix_01::seg000",
    "task_id":    "matrix_01",
    "labels":     ["opponent_modeling", "payoff_analysis"],
    "status":     "ok" | "redacted_ok" | "parse_failed" | "api_failed" |
                  "incomplete_coverage_fallback",
    "error":      Optional[str],
    "redacted":   bool,
    "region":     "thinking" | "answer",
    "chain_truncated": bool,
    "annotated_at": iso8601,
    "batch_mode": "chain",  # so downstream can tell how labels were produced
  }

Usage (calibration, to test chain-mode quality against gold):
  python annotate_chains.py \\
      --segments gold_segments.jsonl \\
      --output   gold_model_labels_chainmode.jsonl \\
      --concurrency 8

Usage (full Phase 1):
  python annotate_chains.py \\
      --segments r1_qwen14b_segments.jsonl \\
      --output   r1_qwen14b_model_labels_chainmode.jsonl \\
      --concurrency 8

Important: for the calibration run, we send ENTIRE chains even if only some
segments are in `gold_segments.jsonl`. This is because whole-chain annotation
requires the full chain context. We reconstruct the chains from the
underlying segments JSONL (r1_qwen14b_segments.jsonl) using the task_ids in
the gold subset. Pass --full_segments to point at the source-of-truth
segments JSONL. If omitted, we assume --segments already contains every
segment of every referenced task (which is true for the full segments file
and false for gold_segments.jsonl).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from openai import AsyncOpenAI
from tqdm import tqdm

from annotation_v2 import (
    annotate_one_chain,
    resolve_api_key,
    ALLOWED_LABELS,
    PROMPT_VERSION,
)

try:
    from google.colab import userdata  # type: ignore
except ImportError:
    userdata = None


logger = logging.getLogger("annotate_chains")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
if not logger.handlers:
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_existing_labels(path: Path) -> Set[str]:
    """For resume: return set of segment_ids already present in output with
    status ok/redacted_ok."""
    done: Set[str] = set()
    if not path.exists():
        return done
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("segment_id")
            if sid and rec.get("status") in ("ok", "redacted_ok"):
                done.add(sid)
    return done


def append_label(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Chain reconstruction
# ---------------------------------------------------------------------------

def build_chains(
    segments: List[Dict],
    full_segments: Optional[List[Dict]] = None,
) -> Dict[str, List[Dict]]:
    """
    Group segments by task_id and sort by seg_index. If full_segments is
    provided, use it as the source of truth for each chain (so chains are
    complete even if `segments` is a subset like gold_segments.jsonl).

    Returns dict: task_id -> sorted list of segment dicts.
    """
    source = full_segments if full_segments is not None else segments

    # Group source segments by task
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for s in source:
        by_task[s["task_id"]].append(s)
    for segs in by_task.values():
        segs.sort(key=lambda x: x["seg_index"])

    # Only keep chains whose task_id appears in `segments`
    keep_tasks = {s["task_id"] for s in segments}
    return {tid: by_task[tid] for tid in keep_tasks if tid in by_task}


def segments_we_need_to_save(segments: List[Dict]) -> Set[str]:
    """Return the set of segment_ids that the caller cares about writing to
    the output file. For full runs this is every segment; for calibration on
    a gold subset, this is just the gold segment_ids."""
    return {s["segment_id"] for s in segments}


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

async def annotate_all_chains(
    client: AsyncOpenAI,
    chains: Dict[str, List[Dict]],
    wanted_segment_ids: Set[str],
    output_path: Path,
    existing: Set[str],
    concurrency: int,
    model: str,
) -> Dict:
    """Send one API call per chain. Write per-segment label records."""
    sem = asyncio.Semaphore(concurrency)
    stats = {
        "chains_total": len(chains),
        "chains_done": 0,
        "chains_failed": 0,
        "segments_written": 0,
        "segments_skipped": 0,
    }

    # Determine which chains have remaining unannotated wanted segments.
    # If every wanted segment in a chain is already in `existing`, skip
    # that chain entirely.
    remaining: List[str] = []
    for tid, segs in chains.items():
        chain_wanted = [s for s in segs if s["segment_id"] in wanted_segment_ids]
        pending = [s for s in chain_wanted if s["segment_id"] not in existing]
        if pending:
            remaining.append(tid)
        else:
            stats["segments_skipped"] += len(chain_wanted)

    logger.info(
        "%d chains to annotate (%d already complete, skipping)",
        len(remaining), len(chains) - len(remaining),
    )

    async def run_one(tid: str) -> None:
        segs = chains[tid]
        # Convert to API payload format: list of {i, region, text} sorted by i
        # (task_id is passed separately)
        payload_segs = [
            {"i": s["seg_index"], "region": s["region"], "text": s["text"]}
            for s in segs
        ]
        async with sem:
            result = await annotate_one_chain(
                client=client,
                task_id=tid,
                chain_segments=payload_segs,
                model=model,
            )

        status = result.get("status", "unknown")
        by_i = result.get("segment_labels", {})
        error = result.get("error")
        redacted = result.get("redacted", False)

        # Write one JSONL row per WANTED segment in this chain
        now = _now_iso()
        for s in segs:
            sid = s["segment_id"]
            if sid not in wanted_segment_ids:
                continue
            if sid in existing:
                continue

            seg_i = s["seg_index"]
            if status in ("ok", "redacted_ok") and seg_i in by_i:
                labs = by_i[seg_i]
                row_status = status
                row_err: Optional[str] = None
            elif status == "incomplete_coverage" and seg_i in by_i:
                # Model returned labels for this index, but missed others.
                # Mark the successful indices with their labels + flag.
                labs = by_i[seg_i]
                row_status = "ok_from_incomplete_chain"
                row_err = (f"chain had incomplete coverage: {error}")
            elif status == "incomplete_coverage":
                # This segment was missed; record the miss.
                labs = []
                row_status = "incomplete_coverage_miss"
                row_err = error
            else:
                labs = []
                row_status = status
                row_err = error

            rec = {
                "segment_id": sid,
                "task_id": tid,
                "labels": labs,
                "status": row_status,
                "error": row_err,
                "redacted": redacted,
                "region": s["region"],
                "chain_truncated": s.get("chain_truncated", False),
                "annotated_at": now,
                "batch_mode": "chain",
            }
            append_label(output_path, rec)
            if labs:
                stats["segments_written"] += 1
            existing.add(sid)  # update so resumption works if we crash

        if status in ("ok", "redacted_ok"):
            stats["chains_done"] += 1
        else:
            stats["chains_failed"] += 1

    tasks = [asyncio.create_task(run_one(tid)) for tid in remaining]
    pbar = tqdm(total=len(tasks), desc="Annotating chains")
    for fut in asyncio.as_completed(tasks):
        await fut
        pbar.update(1)
    pbar.close()

    return stats


async def amain() -> None:
    ap = argparse.ArgumentParser(description="Whole-chain annotator (one API call per chain).")
    ap.add_argument("--segments", type=str, required=True,
                    help="JSONL of segments to annotate. If this is a subset "
                         "(e.g. gold), use --full_segments to provide the "
                         "complete chain context.")
    ap.add_argument("--full_segments", type=str, default=None,
                    help="If --segments is a subset, path to the full "
                         "segments JSONL for reconstructing complete chains.")
    ap.add_argument("--output", type=str, required=True,
                    help="JSONL output path, one row per segment.")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Concurrent chains in flight.")
    ap.add_argument("--gpt_model", type=str, default="gpt-5.2")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    api_key = resolve_api_key(args.api_key)
    client = AsyncOpenAI(api_key=api_key)

    segments = load_jsonl(Path(args.segments))
    full_segments = load_jsonl(Path(args.full_segments)) if args.full_segments else None

    logger.info(
        "loaded %d segments from %s (full_segments=%s)",
        len(segments), args.segments,
        f"{len(full_segments)} from {args.full_segments}" if full_segments else "None",
    )
    logger.info("prompt_version=%s_chainmode", PROMPT_VERSION)

    chains = build_chains(segments, full_segments=full_segments)
    wanted = segments_we_need_to_save(segments)
    logger.info(
        "grouped into %d chains; %d wanted segments",
        len(chains), len(wanted),
    )

    out_path = Path(args.output)
    if args.no_resume and out_path.exists():
        out_path.unlink()
    existing = load_existing_labels(out_path)
    if existing:
        logger.info("resume: %d segments already labeled", len(existing))

    stats = await annotate_all_chains(
        client=client,
        chains=chains,
        wanted_segment_ids=wanted,
        output_path=out_path,
        existing=existing,
        concurrency=args.concurrency,
        model=args.gpt_model,
    )

    logger.info(
        "done: chains_done=%d chains_failed=%d segments_written=%d "
        "segments_skipped=%d",
        stats["chains_done"], stats["chains_failed"],
        stats["segments_written"], stats["segments_skipped"],
    )
    logger.info("output: %s", out_path)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
