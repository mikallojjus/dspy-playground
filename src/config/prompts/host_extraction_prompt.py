HOST_EXTRACTION_PROMPT="""
You are an expert at extracting host information from podcasts. 

You are given podcast episode titles, descriptions, and a truncated transcript.
Extract ONLY the hosts (people conducting the interview), NOT the guests.

Rules:
- Extract FULL NAMES ONLY (e.g., "John Smith", not just "John")
- Do not include titles etc. just have the full name
- Ignore partial names, first names only, or incomplete references
- Only extract hosts (people conducting interviews), NOT guests (people being interviewed)
- Look for host introductions: "I'm...", "Welcome to my show...", "This is [Name] and..."
- Look for speaker patterns in transcript: "[Name]:" at the start of lines
- Hosts typically introduce themselves and welcome guests
- Include any associated URLs (Twitter/X, LinkedIn, Instagram, personal websites)
- URLs must be full links (e.g., https://twitter.com/username)
- If no URLs for a host, use an empty list
- Output as JSON as string (not in markdown block): {{"hosts": [{{"name": "Full Name", "urls": ["url1"]}}]}}
- Do not include any other text
Episode:
title: {title}
description: {description}
transcript: {truncated_transcript}
""".strip()
