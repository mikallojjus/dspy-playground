"""Score claims.judge_equivalence against tests/fixtures/claim_equivalence_eval.json.

Live (real Gemini): GEMINI_API_KEY=… uv run python scripts/eval_claim_equivalence.py [--pairs N]
Prints one line per pair with the verdict and rationale, then precision/recall for `equivalent`.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HATCHET_CLIENT_TOKEN", "x")
os.environ.setdefault("HATCHET_CLIENT_HOST_PORT", "x")

from src.extraction.claim_equivalence_judge import ClaimEquivalenceJudge  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "claim_equivalence_eval.json"


async def main() -> int:
    fixture = json.loads(FIXTURE.read_text())
    pairs, batches = fixture["pairs"], fixture.get("batches", [])
    judge = ClaimEquivalenceJudge()
    print(f"model={judge.model_name} pairs={len(pairs)} batches={len(batches)}")
    tp = fp = fn = tn = unsure = 0
    started = time.time()
    for pair in pairs:
        verdict = (await judge.judge(pair["claim"], [pair["candidate"]]))[0]
        got = verdict.verdict
        accepted = pair["expected"] if isinstance(pair["expected"], list) else [pair["expected"]]
        want = accepted[0]
        if got in accepted and len(accepted) > 1:
            mark = "ok~ "  # borderline pair, either verdict accepted
        elif got == "unsure":
            unsure += 1
            mark = "?   "
        elif got == "equivalent" and want == "equivalent":
            tp += 1
            mark = "ok  "
        elif got == "equivalent":
            fp += 1
            mark = "MISS"
        elif want == "equivalent":
            fn += 1
            mark = "MISS"
        else:
            tn += 1
            mark = "ok  "
        print(f"{mark} {pair['id']:<28} want={'/'.join(accepted):<27} got={got:<14} | {verdict.rationale[:110]}")
    # Batched, as production calls the judge: one call per claim with its whole candidate list.
    batch_misses = 0
    for batch in batches:
        verdicts = await judge.judge(batch["claim"], [c["text"] for c in batch["candidates"]])
        equivalent = {c["id"] for c, v in zip(batch["candidates"], verdicts) if v.verdict == "equivalent"}
        want = set(batch["expected_equivalent_ids"])
        ok = equivalent == want
        batch_misses += 0 if ok else 1
        print(f"{'ok  ' if ok else 'MISS'} {batch['id']:<28} equivalent={sorted(i[:8] for i in equivalent)} want={sorted(i[:8] for i in want)}")
        for c, v in zip(batch["candidates"], verdicts):
            if v.verdict != "not_equivalent":
                print(f"       {v.verdict:<14} {c['id'][:8]} | {c['text'][:60]} | {v.rationale[:90]}")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(
        f"\nequivalent: precision={precision:.2f} recall={recall:.2f} "
        f"(tp={tp} fp={fp} fn={fn} tn={tn} unsure={unsure}); batches missed={batch_misses}; {time.time() - started:.0f}s"
    )
    return 0 if fp == 0 and fn == 0 and batch_misses == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
