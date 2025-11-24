KEYWORD_EXTRACTION_PROMPT="""You are an expert at analyzing podcast episodes and extracting relevant information.

Available Topics (select from this list for topics field):
{topics_list}

Your task has TWO parts:

PART 1 - Generate Keywords (Free-form):
Generate {min_keywords}-{max_keywords} NEW keywords that capture the main themes and concepts discussed in the episode.
- Create NEW keywords based on episode content (do NOT limit to topics list above)
- Use sentence case capitalization (first letter capitalized, rest lowercase unless proper noun)
- Examples of good keywords: "Neural networks", "Quantum computing", "Climate policy", "Economic growth", "Military strategy", "Game design", "Cryptocurrency trading"
- Focus on specific concepts, technologies, people, events, or ideas discussed
- Keywords should be searchable and descriptive

PART 2 - Select Topics (From provided list):
Select {min_topics}-{max_topics} topics from the Available Topics list above that are relevant to the episode.
- ONLY use topics from the provided list above
- Use exact capitalization as shown in the list
- Select the most relevant broad categories
- If no topics are clearly relevant, you may select 0 topics

Output Format:
Return ONLY valid JSON without markdown block in this format:
{{"keywords": ["Keyword One", "Keyword Two", "Keyword Three"], "topics": ["Topic from list", "Another topic from list"]}}

Be precise, relevant, and follow the exact JSON format."""