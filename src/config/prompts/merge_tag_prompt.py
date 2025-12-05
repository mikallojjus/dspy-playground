MERGE_TAG_PROMPT = """CRITICAL: Return ONLY the exact text from the lists below. Do not modify, paraphrase, or change ANY text.

Task: For each keyword, identify which candidates mean EXACTLY the same thing (not just similar/related).

Exact synonym = identical meaning, just different formatting
Examples of EXACT synonyms:
  - "Russia-Ukraine war" = "Russia - Ukraine war" (punctuation difference)
  - "María Corina Machado" = "Maria Corina Machado" (accent difference)
  - "AI policy" = "Artificial Intelligence policy" (abbreviation expansion)

Examples of NOT exact (related but different):
  - "AI" ≠ "AI policy" (different scope - one is broader)
  - "Gaza war" ≠ "Gaza conflict" (different terms - war vs conflict)
  - "Ukraine War" ≠ "War in Ukraine" (different framing)

Input (copy these EXACTLY):

{input_section}

Output format - return ONLY valid JSON array:
[
  {{"keyword": "exact keyword from above", "exact_synonyms": ["candidate1", "candidate2"]}},
  {{"keyword": "another keyword", "exact_synonyms": []}},
  ...
]

Rules:
- Copy exact text only (including punctuation, capitalization, accents)
- If no exact matches, use empty array: []
- Return valid JSON only
- Do not add explanations or comments
- Each keyword must appear in output exactly once"""