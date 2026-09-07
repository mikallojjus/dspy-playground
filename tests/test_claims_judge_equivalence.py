"""claims.judge_equivalence without Gemini: the prompt contract, verdict assembly, the
missing-verdict fallback, input bounds, and registry wiring."""

import pytest

from src.api.schemas.claims_judge_equivalence_schema import ClaimsJudgeEquivalenceInput, ClaimText
from src.extraction.claim_equivalence_judge import LLMAssessment, LLMVerdict, build_prompt, verdict_from
from src.tasks.claims_judge_equivalence import assemble


def test_prompt_makes_the_model_do_the_two_directional_test_and_indexes_candidates():
    p = build_prompt(
        "Sunlight exposure can cause skin cancer.",
        ["Sun exposure may cause skin cancer.", "Sunlight is good for you."],
    )
    assert "same claim: any situation that makes one true makes the other true" in p
    assert "if the CLAIM is true, MUST the CANDIDATE be true" in p
    assert "if the CANDIDATE is true, MUST the\n   CLAIM be true" in p
    assert "same topic, same policy, same side of the debate" in p
    assert "entails the other but not the reverse" in p
    assert "Judge each candidate on its own" in p
    assert "[0] Sun exposure may cause skin cancer." in p and "[1] Sunlight is good for you." in p
    assert "CLAIM:\nSunlight exposure can cause skin cancer." in p


def _assessment(**overrides) -> LLMAssessment:
    base = dict(
        candidate_index=0,
        claim_asserts="an effect",
        candidate_asserts="an effect",
        claim_implies_candidate=True,
        candidate_implies_claim=True,
        relation="same",
        decisive_difference="none",
    )
    return LLMAssessment(**{**base, **overrides})


def test_verdict_is_derived_from_the_directional_answers_not_taken_from_the_model():
    # Both directions and `same`: equivalent.
    assert verdict_from(_assessment()).verdict == "equivalent"
    # The model affirming both directions but naming a different relation is a difference it
    # found itself — related claims on the same side are the failure this guards against.
    related = verdict_from(
        _assessment(candidate_asserts="a purpose", relation="related", decisive_difference="signal vs purpose")
    )
    assert related.verdict == "not_equivalent"
    assert related.rationale == "an effect vs a purpose: signal vs purpose"
    # One direction missing: not equivalent, whatever the relation says.
    assert verdict_from(_assessment(candidate_implies_claim=False)).verdict == "not_equivalent"
    assert verdict_from(_assessment(claim_implies_candidate=False)).verdict == "not_equivalent"
    # Unsure wins over everything.
    assert verdict_from(_assessment(unsure=True)).verdict == "unsure"
    # An unexplained non-equivalence still gets a readable rationale.
    bare = verdict_from(_assessment(relation="candidate_implies_claim_only", decisive_difference=""))
    assert bare.rationale.endswith("candidate implies claim only")


def test_assemble_surfaces_equivalent_and_unsure_and_keeps_every_verdict_in_order():
    inp = ClaimsJudgeEquivalenceInput(
        claim=ClaimText(id="new", text="X"),
        candidates=[ClaimText(id="a", text="A"), ClaimText(id="b", text="B"), ClaimText(text="C")],
    )
    verdicts = [
        LLMVerdict(candidate_index=0, verdict="equivalent", rationale="same proposition"),
        LLMVerdict(candidate_index=1, verdict="not_equivalent", rationale="adds a quantity"),
        LLMVerdict(candidate_index=2, verdict="unsure", rationale="ambiguous referent"),
    ]
    out = assemble(inp, verdicts, "fake-model")
    assert [j.verdict for j in out.judged] == ["equivalent", "not_equivalent", "unsure"]
    assert [j.index for j in out.judged] == [0, 1, 2]
    assert [j.id for j in out.equivalent] == ["a"] and out.equivalent[0].rationale == "same proposition"
    assert [j.text for j in out.unsure] == ["C"] and out.unsure[0].id is None
    assert out.claim.id == "new" and out.model_used == "fake-model"


@pytest.mark.asyncio
async def test_judge_fills_skipped_indices_with_unsure(monkeypatch):
    from src.extraction import claim_equivalence_judge as mod

    class Fake(mod.ClaimEquivalenceJudge):
        def __init__(self):  # no client, no key
            self.model_name = "fake"

        async def _call_gemini(self, prompt: str) -> str:
            # the model answers only for index 1 and invents an out-of-range index
            same = (
                '"claim_asserts": "x", "candidate_asserts": "x", "claim_implies_candidate": true, '
                '"candidate_implies_claim": true, "relation": "same", "decisive_difference": "none"'
            )
            return (
                '{"assessments": [{"candidate_index": 1, ' + same + '}, {"candidate_index": 7, ' + same + '}]}'
            )

    verdicts = await Fake().judge("X", ["A", "B", "C"])
    assert [v.verdict for v in verdicts] == ["unsure", "equivalent", "unsure"]
    assert verdicts[0].rationale == "no verdict returned"
    assert await Fake().judge("X", []) == []


def test_registry_has_the_task_with_its_contract():
    from src.api.schemas.claims_judge_equivalence_schema import ClaimsJudgeEquivalenceResult
    from src.tasks.claims_judge_equivalence import CLAIMS_JUDGE_EQUIVALENCE_SPEC
    from src.tasks.registry import get_task

    entry = get_task("claims.judge_equivalence")
    assert entry is not None
    assert entry.input_model is ClaimsJudgeEquivalenceInput
    assert entry.output_model is ClaimsJudgeEquivalenceResult
    assert CLAIMS_JUDGE_EQUIVALENCE_SPEC.rate_limit_key == "gemini_global"
    assert get_task("claims.judge_duplicates") is None  # the retrieval-coupled name is gone


def test_input_bounds():
    with pytest.raises(ValueError):
        ClaimsJudgeEquivalenceInput(claim=ClaimText(text="x"), candidates=[])
    with pytest.raises(ValueError):
        ClaimsJudgeEquivalenceInput(claim=ClaimText(text=""), candidates=[ClaimText(text="y")])
    with pytest.raises(ValueError):
        ClaimsJudgeEquivalenceInput(claim=ClaimText(text="x"), candidates=[ClaimText(text="y")] * 51)
