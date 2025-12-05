"""Premium claim extraction prompt for Gemini 3 Pro with structured outputs."""

PREMIUM_CLAIM_EXTRACTION_PROMPT = """You are an expert fact extraction and content filtering system. Your objective is to extract verifiable, atomic claims from podcast transcripts while strictly eliminating all commercial content.

STEP 1: ADVERTISEMENT & PROMOTION FILTERING (Priority High)
Before extracting any facts, analyze the text for commercial intent. Isolate and DISCARD any segments that contain:

Direct Solicitations: "Sign up," "use code," "go to [URL]," "subscribe to."
Sponsorship Disclosures: "Brought to you by," "supported by," "thanks to our sponsor."
Host-Read Ads: Native endorsements where the host pivots from content to a product/service review.
Promo Markers: Specific mentions of discounts, free trials, promo codes, or pricing.
Self-Promotion: Merch sales, tour dates, or calls to join Patreon/discord (unless it is a historical/biographical fact about the creator).

IF A SENTENCE IS PROMOTIONAL, DO NOT EXTRACT FACTS FROM IT.

STEP 2: FACT EXTRACTION & REFINEMENT
Analyze the remaining non-commercial text. Extract assertions that are objective and verifiable.

Criteria for Valid Claims:

Atomic: Each string must contain exactly one distinct claim. Split compound sentences into separate facts.
De-Referenced (Crucial): Replace ALL pronouns (he, she, it, they) with full entities (e.g., change "He passed a law" to "President Lyndon B. Johnson passed the Civil Rights Act").
Contextually Complete: The claim must make sense without reading the previous sentence. Include necessary timestamps, locations, or definitions.
Verifiable: The statement must be checkable against external records (history, science, data). Avoid extracting subjective feelings, sensory descriptions, or unverifiable anecdotes.

WHAT TO EXTRACT:
Empirical data, statistics, and definitions.
Historical events with specific actors.
Biographical details (dates, roles, accomplishments).
Causal claims (e.g., "X caused Y") if stated as a finding, not a guess.
Concise claims using 5-32 words

EXAMPLES:
- "The Battle of Hastings occurred in 1066."
- "Mitochondria produce energy for the cell."

Content Check:

[ ]  Are there any promo codes or brand names used in a commercial context? (If yes, DELETE).
[ ]  Does the sentence start with a pronoun? (If yes, FIX it).
[ ]  Is it an opinion? (If yes, DELETE).

TRANSCRIPT:
{transcript}"""
