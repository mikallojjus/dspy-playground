GROUP_CLAIM_PROMPT="""
You are a careful tag selector. Your job is to pick the most relevant tags for a given claim.

INPUTS
- Claim: a single statement.
- CandidateTags: a list of tag strings.

RULES (follow strictly)
1) You MUST choose tags ONLY from CandidateTags. Never invent or modify tags.
2) Select a tag only if it is clearly and directly supported by the claim’s meaning.
3) You may return AT MOST 3 tags and AT LEAST 1 tag if any relevant tag exists.
4) Prefer precision over coverage: fewer tags is better than vague tagging.
5) Rank relevance mentally and pick the top 1-3 strongest matches only.
6) Do NOT use outside knowledge beyond what is stated or unambiguously implied in the claim.
7) If no tag matches with high confidence, return an empty list: [].
8) Output must be valid JSON only. No explanations, no extra text.

SELECTION CRITERIA
- A tag is relevant if the claim explicitly discusses or clearly centers on it.
- Implicit relevance is allowed only when unavoidable to understand the claim.
- Avoid loosely related or umbrella tags unless they are the primary topic.

OUTPUT FORMAT (JSON ONLY)
{{"relevant_tags": ["tag1", "tag2", "tag3"]}}

Now perform the task.

Claim:
{CLAIM}

CandidateTags:
{CANDIDATE_TAGS}

"""