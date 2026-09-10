"""Offline experiment for the Debate/Simple scope classification — no Hatchet, no API.

The product change is a scope rule inside claims.extract; this is the fast loop that
lets the rule be iterated against a labelled set before it goes anywhere near the
extraction prompt. Classifying fixed claims costs one call per run instead of a full
extraction, so a rubric edit is scored in ~20 seconds.

    uv run python -m src.cli.eval_debatable gold --rubric v1 --runs 3
    uv run python -m src.cli.eval_debatable gold --rubric v1 --rubric v2   # A/B

Rubric source: the Debate tag definition (Category "Form", counterpart "Simple",
verified 2026-09-10). Debate classifies SCOPE, not truth and not whether anyone is
currently arguing about it.
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.api.schemas.claims_extract_schema import ClaimsExtractInput
from src.config.settings import settings
from src.extraction.claims_extractor import ClaimsExtractor
from src.extraction.claims_prompt_builder import build_extract_prompt
from src.pipeline.claims_extract_core import assemble_result

# Profile of the 1,378 claims carrying Geo's `Debate` tag, measured 2026-09-10. The
# target an extraction run is drifting toward or away from.
CORPUS_PROFILE = {
    "median_words": 10.0,
    "normative_pct": 52.3,
    "hedged_pct": 1.9,
    "reports_finding_pct": 0.1,
    "cites_study_pct": 0.3,
}

_SHAPE = {
    "normative_pct": r"\b(should|must|ought to)\b",
    "hedged_pct": r"\b(can|may|could|might)\b",
    "reports_finding_pct": r"\b(found|according to|study (found|shows)|reported|studies have shown)\b",
    "cites_study_pct": r"\b(study|studies|research|peer-reviewed|survey)\b",
}


def shape_metrics(texts: List[str]) -> Dict[str, float]:
    """How far this run sits from the curated Debate corpus. Cheap, deterministic,
    and the first thing to look at after a prompt edit: hedging and length move
    long before the classifier's verdict does."""
    if not texts:
        return {}
    out = {"median_words": statistics.median(len(t.split()) for t in texts)}
    for key, rx in _SHAPE.items():
        out[key] = round(100 * sum(1 for t in texts if re.search(rx, t, re.I)) / len(texts), 1)
    return out

# ── Rubric variants ─────────────────────────────────────────────────────────
# Keep every version that has been scored; the winner is what moves into
# src/config/prompts/claims_extract/sections.py.

_V1 = """Classify each claim's FORM as `debate` or `simple`.

`debate` — a broad, contestable claim whose central assertion supports substantive
positions for and against it, rather than being resolved through direct verification
of one narrowly scoped fact. It may express a policy judgment, comparison,
interpretation, causal thesis, normative position, or broad forecast.

`simple` — the claim's main assertion is settled by directly verifying one narrowly
scoped fact.

Judge the SCOPE of the claim's main assertion. Not whether it is true, and not
whether anyone is currently arguing about it. A claim can be both factual and
`debate` — form and truth are independent.

Rules:
1. Tag the MAIN assertion. Concrete anchors, examples or figures attached to a broad
   claim do not rescue it: "The US has historically used inflation to manage debt,
   such as after WWII" is `debate`.
2. Naming companies or products does not make a claim `simple`; scope decides:
   "OpenAI and Anthropic are the main competitors in frontier AI" is `debate`.
3. Comparative-superiority stances ("X is better than Y", "X is the main competitor
   to Y", "X would choke Y") are `debate`. A factual comparison of named entities on
   a checkable attribute (regulatory status, features, dates) is `simple`.
4. Unattributed economy-wide or market-wide forecasts with no specific market,
   instrument or figure ("a recession is likely next year", "crypto will replace
   banks") are `debate`. A forecast about one specific market, asset or event ("the
   largest wave of commercial real estate debt maturities is expected in the fall")
   is `simple`.
5. Advice, and "an indicator to watch for", state no checkable fact, so they are
   `debate`."""

RUBRICS: Dict[str, str] = {"v1": _V1}

# ── Factuality rubric variants ──────────────────────────────────────────────
# Deliberately asymmetric: a false negative costs one missing flag, a false
# positive publishes an interpretation as a checkable fact and sends readers to
# Verify/Dispute on something no source can settle.

_STRICT_V1 = """Classify each claim as `factual` or `not_factual`.

`factual` — the claim asserts ONE specific, checkable fact that a fact-checker
could confirm or refute against a source: a named actor doing a named thing, a
dated event, a quantity or measurement, a location, an attributable on-record
statement, or a specific named study's finding.

`not_factual` — everything else. In particular:
- evaluations, value judgments and appraisals of magnitude ("significant",
  "insufficient", "too", "better", "risky")
- normative or policy positions (should, must, needs to)
- interpretations, characterisations and causal theses
- generalisations with no specific referent ("many people", "most", "often",
  "some", "not everyone")
- hedged statements (may, might, can, could, tends to, is likely to)
- forecasts and predictions
- definitional or descriptive statements with nothing to check

Truth is not the question: a specific claim that turns out to be wrong is still
`factual`. Specificity and checkability are the question.

A false negative is cheap; a false positive is not. If you are not certain the
claim is specifically checkable against a source, answer `not_factual`."""

FACT_RUBRICS: Dict[str, str] = {"strict_v1": _STRICT_V1}

# strict_v1 collapsed: told "if not certain, answer not_factual", the model answered
# not_factual for all 30, including "Tyler Robinson killed Charlie Kirk". Two causes —
# the asymmetry had no counterweight, and "checkable" was read as "I can verify it
# myself". v2 keeps the asymmetry, anchors both classes with worked examples, and
# separates checkability from the model's own knowledge.

_STRICT_V2 = """Classify each claim as `factual` or `not_factual`.

The question is what KIND of assertion the claim makes, not whether it is true and
not whether you happen to know the answer.

`factual` — the claim asserts one specific thing about the world that a source
could settle: a named actor doing a named thing, a dated event, a quantity or
measurement, a location, a named study's finding, or an on-record statement by a
named person. You do not need to know whether it is correct, and you do not need
to be able to check it yourself. A specific claim that turns out to be false is
still `factual`.

Examples of `factual`:
- "Tyler Robinson killed Charlie Kirk." (named actor, named act)
- "A 2023 University of Exeter study of 450,000 participants found morning types
  had a lower risk of depression." (named study, sample, stated finding)
- "Gaza Strip has been subjected to a continuous blockade since June 2007." (dated)
- "Sam Altman expects an internal OpenAI system he would call AGI by end of 2026."
  (on-record statement by a named person)

`not_factual` — the claim has no single specific thing a source could settle:
- evaluations and appraisals of magnitude: "significant", "insufficient", "too",
  "risky", "low-cost", "better"
- normative positions: should, must, needs to
- interpretations, characterisations and causal theses without a named study
- generalisations with no specific referent: "many people", "most", "some",
  "often", "not everyone"
- hedged statements: may, might, can, could, tends to, is likely to
- forecasts and predictions
- definitional or descriptive statements with nothing to verify

Examples of `not_factual`:
- "Antidepressants are overprescribed." (evaluative)
- "For some people, AI chatbots can increase their social abilities." (hedged, no
  specific referent)
- "Flock Safety cameras reduce crime." (causal thesis, no study named)
- "AI will create more jobs than it eliminates." (forecast)

Both labels will occur. Classify each claim on its own terms; do not aim for any
particular balance. When a claim genuinely sits between the two — it names
something specific but wraps it in an appraisal — choose `not_factual`, because a
missing flag costs less than a wrong one."""

FACT_RUBRICS["strict_v2"] = _STRICT_V2

TASKS = {
    "scope": (RUBRICS, ("debate", "simple"), "simple",
              "You are classifying the form of extracted claims for a knowledge graph."),
    "factual": (FACT_RUBRICS, ("factual", "not_factual"), "not_factual",
                "You are classifying whether extracted claims assert specific checkable facts."),
}


class ScopeLabel(BaseModel):
    index: int = Field(description="0-based index of the claim being classified.")
    # Task-agnostic on purpose. This description is part of the decode-constrained
    # schema, so naming one task's labels here silently forces every other task's
    # answers into the fallback — which reads as a confident, uniform verdict.
    label: str = Field(description="Exactly one of the labels defined in the instructions.")


class ScopeLabels(BaseModel):
    labels: List[ScopeLabel] = Field(default_factory=list)


def build_prompt(rubric: str, claims: List[str], task: str = "scope") -> str:
    header = TASKS[task][3]
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return (
        f"{header}\n\n"
        f"{rubric}\n\n"
        "Return one label per claim, by index. Every claim gets exactly one label.\n\n"
        f"CLAIMS\n{listing}"
    )


async def classify(
    extractor: ClaimsExtractor, rubric: str, claims: List[str], task: str = "scope"
) -> List[str]:
    """One batched call. Anything the model omits or mislabels falls to the task's
    conservative label — a claim we cannot confidently call broad should not become a
    debate motion, and one we cannot confidently call checkable should not be flagged
    factual."""
    _, allowed, fallback, _ = TASKS[task]
    text = await extractor._call_gemini(
        prompt=build_prompt(rubric, claims, task),
        config=extractor._config(ScopeLabels),
        step_name=f"{task} classification",
    )
    parsed = ScopeLabels.model_validate_json(text)
    out = [fallback] * len(claims)
    recognised = 0
    for row in parsed.labels:
        if 0 <= row.index < len(claims) and row.label in allowed:
            out[row.index] = row.label
            recognised += 1
    # A run where most answers fell through to the fallback is instrumentation
    # failure, not a finding. Say so loudly rather than reporting a clean collapse.
    if recognised < 0.8 * len(claims):
        seen = sorted({row.label for row in parsed.labels})
        raise RuntimeError(
            f"{task}: only {recognised}/{len(claims)} labels were recognised "
            f"(allowed {allowed}, model returned {seen}) — the fallback would have "
            f"produced a uniform result"
        )
    return out


async def score_rubric(
    extractor: ClaimsExtractor, name: str, rubric: str, gold: List[Dict[str, Any]], runs: int,
    task: str = "scope",
) -> Dict[str, Any]:
    """Shuffle per run so a batch cannot be answered by balancing the classes, then
    take the majority label per claim across runs."""
    votes: List[List[str]] = [[] for _ in gold]
    accuracies: List[float] = []
    for run in range(runs):
        order = list(range(len(gold)))
        random.Random(1000 + run).shuffle(order)
        labels = await classify(extractor, rubric, [gold[i]["text"] for i in order], task)
        correct = 0
        for slot, gold_index in enumerate(order):
            votes[gold_index].append(labels[slot])
            if labels[slot] == gold[gold_index]["label"]:
                correct += 1
        accuracies.append(correct / len(gold))
        print(f"  {name} run {run + 1}: accuracy {100 * accuracies[-1]:.1f}%")

    positive = TASKS[task][1][0]
    majority = [max(set(v), key=v.count) for v in votes]
    tp = sum(1 for g, m in zip(gold, majority) if g["label"] == positive and m == positive)
    fp = sum(1 for g, m in zip(gold, majority) if g["label"] != positive and m == positive)
    fn = sum(1 for g, m in zip(gold, majority) if g["label"] == positive and m != positive)
    tn = sum(1 for g, m in zip(gold, majority) if g["label"] != positive and m != positive)
    unstable = sum(1 for v in votes if len(set(v)) > 1)
    return {
        "rubric": name,
        "runs": runs,
        "accuracy_mean": round(statistics.mean(accuracies), 3),
        "accuracy_runs": [round(a, 3) for a in accuracies],
        # Precision on `debate` is the number that matters: a false positive publishes a
        # narrow fact as a debate motion, which is the failure this feature exists to stop.
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "unstable_claims": unstable,
        "majority": majority,
    }


async def cmd_gold(args: argparse.Namespace) -> None:
    gold = json.loads(Path(args.gold).read_text())
    extractor = ClaimsExtractor()
    results = []
    for name in args.rubric:
        print(f"\n=== [{args.task}] rubric {name} ({len(gold)} gold claims, {args.runs} runs) ===")
        results.append(await score_rubric(extractor, name, TASKS[args.task][0][name], gold, args.runs, args.task))

    for r in results:
        print(f"\n--- {r['rubric']}: accuracy {100 * r['accuracy_mean']:.1f}%  "
              f"precision {r['precision']}  recall {r['recall']}  "
              f"unstable {r['unstable_claims']}")
        print(f"    confusion {r['confusion']}")
        print("    disagreements (gold -> predicted):")
        for g, m in zip(gold, r["majority"]):
            if g["label"] != m:
                print(f"      [{g['label']:6} -> {m:6}] ({g['source']}) {g['text'][:96]}")

    if args.hard:
        hard = json.loads(Path(args.hard).read_text())
        labels = await classify(extractor, TASKS[args.task][0][args.rubric[-1]], [h["text"] for h in hard], args.task)
        print("\n--- held-out borderline cases (no gold label; for judgement, not scoring)")
        for h, label in zip(hard, labels):
            print(f"      [{label:6}] {h['text'][:88]}  <- {h['note']}")

    if args.out:
        Path(args.out).write_text(json.dumps({"gold_size": len(gold), "results": results}, indent=1))
        print(f"\nwrote {args.out}")


async def cmd_pipeline(args: argparse.Namespace) -> None:
    """Extract from real debate transcripts, then classify every claim. Answers the
    question a rubric alone cannot: how much of what we generate is publishable as a
    debate motion, and what exactly gets dropped."""
    extractor = ClaimsExtractor()
    rubric = RUBRICS[args.rubric]
    report: List[Dict[str, Any]] = []

    for path in args.inputs:
        spec = json.loads(Path(path).read_text())
        inp = ClaimsExtractInput(**spec["payload"])
        extraction = await extractor.extract_claims(build_extract_prompt(inp, []))
        result = assemble_result(inp, extraction.model_dump(), [], model_used=settings.claims_extract_model)
        claims = [{"text": c.text, "is_factual": c.is_factual,
                   "prompt_contestable": c.is_contestable} for c in result.claims]
        labels = await classify(extractor, rubric, [c["text"] for c in claims]) if claims else []
        # Second opinion on factuality: what the strict rubric would flag, next to what
        # the extraction prompt itself flagged. The gap is the size of the prompt change.
        facts = await classify(extractor, FACT_RUBRICS[args.fact_rubric],
                               [c["text"] for c in claims], "factual") if claims else []
        for c, label, fact in zip(claims, labels, facts):
            c["scope"] = label
            c["strict_factual"] = fact == "factual"
        kept = [c for c in claims if c["scope"] == "debate"]
        report.append({"name": spec["name"], "motion": spec.get("motion"), "claims": claims,
                       "kept": len(kept), "total": len(claims),
                       "shape_all": shape_metrics([c["text"] for c in claims]),
                       "shape_kept": shape_metrics([c["text"] for c in kept])})
        print(f"  {spec['name']}: {len(kept)}/{len(claims)} survive the debate filter")

    print("\n=== corpus profile (target = Geo's 1,378 Debate-tagged claims) ===")
    everything = [c["text"] for r in report for c in r["claims"]]
    survivors = [c["text"] for r in report for c in r["claims"] if c["scope"] == "debate"]
    allm, keptm = shape_metrics(everything), shape_metrics(survivors)
    print(f"  {'metric':22} {'target':>8} {'extracted':>10} {'survivors':>10}")
    for k, target in CORPUS_PROFILE.items():
        print(f"  {k:22} {target:>8} {allm.get(k, 0):>10} {keptm.get(k, 0):>10}")

    total = sum(r["total"] for r in report)
    kept = sum(r["kept"] for r in report)
    from collections import Counter
    fact_all = Counter(str(c["is_factual"]) for r in report for c in r["claims"])
    fact_kept = Counter(str(c["is_factual"]) for r in report for c in r["claims"] if c["scope"] == "debate")
    print(f"\n  survival: {kept}/{total} ({100 * kept / total:.0f}%)")
    print(f"  is_factual all       : {dict(fact_all)}")
    print(f"  is_factual survivors : {dict(fact_kept)}   (Geo corpus: 4% of flagged are true)")
    surv = [c for r in report for c in r["claims"] if c["scope"] == "debate"]
    strict_true = sum(1 for c in surv if c["strict_factual"])
    prompt_true = sum(1 for c in surv if c["is_factual"])
    flipped = [c for c in surv if c["is_factual"] and not c["strict_factual"]]
    # Does the rubric still hold once embedded in the full extraction prompt, rather
    # than run standalone over fixed text? Agreement is a SIGNAL, not a score: the
    # standalone classifier was validated on a hand-judged gold set, which is a
    # different distribution from raw extraction output, and on inspection it is the
    # one that errs on some disputes (it misses rule 3 comparatives). Read the
    # disagreements and judge them; do not treat either side as ground truth.
    scored = [c for r in report for c in r["claims"] if c["prompt_contestable"] is not None]
    if scored:
        agree = sum(1 for c in scored if c["prompt_contestable"] == (c["scope"] == "debate"))
        pc = sum(1 for c in scored if c["prompt_contestable"])
        sc = sum(1 for c in scored if c["scope"] == "debate")
        print(f"\n  contestable — in-prompt: {pc}/{len(scored)}  standalone rubric: {sc}/{len(scored)}"
              f"  agreement {100 * agree / len(scored):.0f}%")
        for c in scored:
            if c["prompt_contestable"] != (c["scope"] == "debate"):
                print(f"      in-prompt={c['prompt_contestable']!s:5} standalone={c['scope']:6} {c['text'][:82]}")
    print(f"\n  survivors flagged factual — extraction prompt: {prompt_true}/{len(surv)} "
          f"({100 * prompt_true / len(surv):.0f}%)")
    print(f"  survivors flagged factual — strict rubric:     {strict_true}/{len(surv)} "
          f"({100 * strict_true / len(surv):.0f}%)")
    print(f"  the prompt over-flags {len(flipped)} of them:")
    for c in flipped[:14]:
        print(f"      {c['text'][:104]}")

    print("\n=== DROPPED as `simple` ===")
    for r in report:
        for c in r["claims"]:
            if c["scope"] != "debate":
                print(f"  [{str(c['is_factual'])[:5]:5}] {c['text'][:110]}")
    print("\n=== KEPT as `debate` ===")
    for r in report:
        for c in r["claims"]:
            if c["scope"] == "debate":
                print(f"  [{str(c['is_factual'])[:5]:5}] {c['text'][:110]}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.out}")


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--gold", default="/private/tmp/claude-501/eval-debatable/gold.json")
    g.add_argument("--hard", default=None)
    g.add_argument("--task", default="scope", choices=sorted(TASKS))
    g.add_argument("--rubric", action="append", default=None)
    g.add_argument("--runs", type=int, default=3)
    g.add_argument("--out", default=None)
    r = sub.add_parser("pipeline")
    r.add_argument("inputs", nargs="+")
    r.add_argument("--rubric", default="v1", choices=sorted(RUBRICS))
    r.add_argument("--fact-rubric", dest="fact_rubric", default="strict_v2", choices=sorted(FACT_RUBRICS))
    r.add_argument("--out", default=None)

    args = p.parse_args(argv)
    print(f"model {settings.claims_extract_model} @ temp {settings.claims_extract_temperature}")
    if args.cmd == "pipeline":
        asyncio.run(cmd_pipeline(args))
    else:
        args.rubric = args.rubric or [sorted(TASKS[args.task][0])[0]]
        asyncio.run(cmd_gold(args))


if __name__ == "__main__":
    main(sys.argv[1:])
