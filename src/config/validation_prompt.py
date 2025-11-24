"""
Validation prompt configuration for Gemini claim validation.

Update the VALIDATION_PROMPT constant below with your custom validation logic.
"""

VALIDATION_PROMPT = """
You are evaluating a list of claims taken from podcast transcripts. For each claim, determine whether it is "good" or "bad" based on whether it satisfies all the criteria below.

Good claim definition (all must be met)

Self-contained: Understandable on its own. No missing context, placeholders, or fragmentary language.

No ads: Endorsements, product mentions, and promotional copy are automatically bad.

Coherent syntax: Must be a grammatical, well-formed statement.

Normative/opinion OK if testable: Statements with “should”, “requires”, etc., are allowed if specific and evaluable. Belief-attributions are allowed if concise and self-contained.

No trivial/credit claims: Episode credits, listings, or meta information are not valid claims.

Border guidance

Mark bad: under-specified actors/entities, vague references, ads, credits, anecdotes.

Mark good: clear, specific, self-contained factual or testable statements.

Output rule

For each claim, set is_valid to false if the claim is bad, or true if the claim is good.
"""
