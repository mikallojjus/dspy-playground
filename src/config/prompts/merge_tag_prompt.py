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


TAG_CHECKER_PROMPT = """
You are a strict topic-label checker for a taxonomy.

You will receive a JSON array of **topic labels** (strings).
For each label, check whether it follows all rules below.

Return only JSON: a single object where each key is the original label exactly as given, and the value is an object with:

* is_valid - boolean
* suggested_alternatives  - array of strings

### Rules

1. One concept per topic (no “X and Y” unless a fixed phrase like “Research and development”).
2. Use canonical name (Wikipedia-style; full names for people/orgs).
3. Disambiguate explicitly if ambiguous: `Term (Qualifier)`.
4. Consistent structure (e.g., `Country + domain`, `Domain + subdomain`).
5. Neutral and descriptive (no stance/claim framing).
6. No questions or sentences; short noun phrase only.
7. Prefer full form over acronyms (except globally obvious like `US`).
8. Sentence case.
9. Reusable concepts only (no episode-specific metadata like guest/date/show).

### Output format (JSON only)

{{
  "some label here": {{
    "is_valid": true,
    "suggested_alternatives": []
  }},
  "another label": {{
    "is_valid": false,
    "suggested_alternatives": ["...", "...", "..."]
  }}
}}

### Output rules

* Keys must match the input labels exactly** (same casing, spacing, punctuation).
* If a label passes all rules: `is_valid=true` and `suggested_alternatives=[]`.
* If a label fails any rule: `is_valid=false` and `suggested_alternatives` must contain **3–8** compliant replacement labels. 
* The replacement don't need to specify anything in brackets for example it should not be Comedy (Genre) but just Comedy
* If multiple concepts are present, alternatives may include **split concepts** as separate labels.
* If ambiguity exists, include **disambiguated** alternatives instead of guessing.
* If it's peoples name don't need to change it to People as replacement but check if it's in the correct format no need to change like that. If not in correct format then only update the format.
* Use sentence case in all alternatives.
* Do not invent episode-specific details.
* Do not output anything except JSON.

### Input
{labels_arr}
"""