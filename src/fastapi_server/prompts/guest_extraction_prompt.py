GUEST_EXTRACTION_PROMPT="""
You are given podcast episode titles and descriptions.
Extract ONLY the guests (people being interviewed), NOT the hosts/podcasters.

Rules:
- Extract FULL NAMES ONLY (e.g., "John Smith", not just "John")
- Do not include titles etc. just have the full name
- Ignore partial names, first names only, or incomplete references
- Only extract guests (people being interviewed), NOT hosts and co-hosts (people conducting interviews)
- Look for clear introductions: "Today's guest is...", "We're joined by...", "Interviewing..."
- Hosts often introduce themselves at the start - do NOT extract them
- Include any associated URLs (Twitter/X, LinkedIn, Instagram, personal websites)
- URLs must be full links (e.g., https://twitter.com/username)
- If no URLs for a guest, use an empty list
- Output as JSON as string (not in markdown block): {{"guests": [{{"name": "Full Name", "urls": ["url1"]}}]}}
- Do not include any other text
Episode:
title: {title}
description: {description}
""".strip()