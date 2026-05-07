"""
Segment-oriented annotator runner.

Consumes a JSONL of segments (from segment_chains.py or sample_gold_segments.py)
and emits a JSONL of per-segment labels. Each output line is compatible with
evaluate_gold.py.

Unlike annotation_v2.py (which is chain-oriented and produces a big blob
matching the original chains structure), this tool treats every segment as
an independent annotation unit and writes one line per segment.

Output line schema:
  {
    "segment_id":  "matrix_01::seg000",
    "task_id":     "matrix_01",
    "labels":      ["opponent_modeling", "payoff_analysis"],
    "status":      "ok" | "redacted_ok" | "parse_failed" | "api_failed",
    "error":       Optional[str],
    "redacted":    bool,
    "region":      "thinking" | "answer",
    "chain_truncated": bool,
    "annotated_at": iso8601
  }

Two modes:
  * Calibration mode: run on the 150 gold segments (cheap, for Phase 0.5).
  * Full mode:        run on the full 14k+ segments (for Phase 1).

Both modes take the same --segments JSONL; the two phases just point at
different files.

Usage (calibration):
  python annotate_segments.py \\
      --segments gold_segments.jsonl \\
      --output   gold_model_labels.jsonl \\
      --task_concurrency 8

Usage (full phase 1):
  python annotate_segments.py \\
      --segments r1_qwen14b_segments.jsonl \\
      --output   r1_qwen14b_model_labels.jsonl \\
      --task_concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from tqdm import tqdm

# Reuse the canonical prompt, schema, and single-segment API call logic.
from annotation_v2 import (
    SYSTEM_PROMPT,
    SegmentLabels,
    annotate_one_segment,
    resolve_api_key,
    ALLOWED_LABELS,
    PROMPT_VERSION,
)

try:
    from google.colab import userdata  # type: ignore
except ImportError:
    userdata = None


logger = logging.getLogger("annotate_segments")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
if not logger.handlers:
    logger.addHandler(_h)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_segments_jsonl(path: Path) -> List[Dict]:
    """Load a segments JSONL (full or gold). Requires prev_text/next_text for
    the context window; these are present in gold_segments and can be computed
    on the fly for full segments (see enrich_full_segments)."""
    out: List[Dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def enrich_full_segments(segments: List[Dict]) -> List[Dict]:
    """If segments lack prev_text/next_text (i.e. came from segment_chains.py,
    not sample_gold_segments.py), compute them here by grouping on task_id
    and seg_index."""
    if all("prev_text" in s for s in segments):
        return segments
    from collections import defaultdict
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for s in segments:
        by_task[s["task_id"]].append(s)
    for segs in by_task.values():
        segs.sort(key=lambda x: x["seg_index"])

    enriched: List[Dict] = []
    for segs in by_task.values():
        n = len(segs)
        for i, s in enumerate(segs):
            prev_text = segs[i - 1]["text"] if i > 0 else ""
            next_text = segs[i + 1]["text"] if i < n - 1 else ""
            enriched.append({**s, "prev_text": prev_text, "next_text": next_text})
    return enriched


def load_existing_labels(path: Path) -> Dict[str, Dict]:
    """For resume: read existing output JSONL and return by segment_id."""
    by_id: Dict[str, Dict] = {}
    if not path.exists():
        return by_id
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
            # Only treat fully-successful segments as complete.
            if sid and rec.get("status") in ("ok", "redacted_ok"):
                by_id[sid] = rec
    return by_id


def append_label(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


async def annotate_all(
    client: AsyncOpenAI,
    segments: List[Dict],
    output_path: Path,
    existing: Dict[str, Dict],
    concurrency: int,
    model: str,
) -> Dict:
    sem = asyncio.Semaphore(concurrency)
    stats = {"total": len(segments), "ok": 0, "parse_failed": 0, "api_failed": 0, "skipped": 0}

    async def run_one(seg: Dict) -> None:
        sid = seg["segment_id"]
        if sid in existing:
            stats["skipped"] += 1
            return
        # annotate_one_segment expects target dict with keys i, region, text
        target = {"i": seg["seg_index"], "region": seg["region"], "text": seg["text"]}
        prev_seg = (
            {"i": seg["seg_index"] - 1, "region": seg["region"], "text": seg["prev_text"]}
            if seg.get("prev_text") else None
        )
        next_seg = (
            {"i": seg["seg_index"] + 1, "region": seg["region"], "text": seg["next_text"]}
            if seg.get("next_text") else None
        )
        async with sem:
            res = await annotate_one_segment(
                client=client,
                target=target,
                prev_seg=prev_seg,
                next_seg=next_seg,
                task_id=seg["task_id"],
                model=model,
            )
        # Shape the output record
        rec = {
            "segment_id": sid,
            "task_id": seg["task_id"],
            "labels": res["labels"],
            "status": res["status"],
            "error": res.get("error"),
            "redacted": res.get("redacted", False),
            "region": seg["region"],
            "chain_truncated": seg.get("chain_truncated", False),
            "annotated_at": _now_iso(),
        }
        append_label(output_path, rec)
        existing[sid] = rec

        s = res["status"]
        if s in ("ok", "redacted_ok"):
            stats["ok"] += 1
        elif s == "parse_failed":
            stats["parse_failed"] += 1
        else:
            stats["api_failed"] += 1

    todo = [s for s in segments if s["segment_id"] not in existing]
    pbar = tqdm(total=len(todo), desc="Annotating segments")
    tasks = [asyncio.create_task(run_one(s)) for s in todo]
    for fut in asyncio.as_completed(tasks):
        await fut
        pbar.update(1)
    pbar.close()

    return stats


async def amain() -> None:
    ap = argparse.ArgumentParser(description="Annotate segments (calibration or full).")
    ap.add_argument("--segments", type=str, required=True,
                    help="JSONL of segments (gold or full).")
    ap.add_argument("--output", type=str, required=True,
                    help="JSONL output path (resumable).")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Concurrent segment annotations.")
    ap.add_argument("--gpt_model", type=str, default="gpt-5.2")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    api_key = resolve_api_key(args.api_key)
    client = AsyncOpenAI(api_key=api_key)

    segments = load_segments_jsonl(Path(args.segments))
    segments = enrich_full_segments(segments)
    logger.info("loaded %d segments from %s", len(segments), args.segments)
    logger.info("prompt_version=%s  labels=%s", PROMPT_VERSION, ALLOWED_LABELS)

    output_path = Path(args.output)
    if args.no_resume and output_path.exists():
        output_path.unlink()
    existing = load_existing_labels(output_path)
    if existing:
        logger.info("resume: %d segments already labeled, skipping those", len(existing))

    stats = await annotate_all(
        client=client,
        segments=segments,
        output_path=output_path,
        existing=existing,
        concurrency=args.concurrency,
        model=args.gpt_model,
    )

    logger.info(
        "done: ok=%d parse_failed=%d api_failed=%d skipped=%d total=%d",
        stats["ok"], stats["parse_failed"], stats["api_failed"],
        stats["skipped"], stats["total"],
    )
    logger.info("output: %s", output_path)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
