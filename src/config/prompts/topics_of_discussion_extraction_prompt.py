TOPICS_OF_DISCUSSION_PROMPT="""
You are an expert podcast content analyst specializing in identifying and labeling topics of discussion from long-form podcast transcripts while preserving their chronological order.

Instructions

Read the provided title, description, and transcript of the podcast episode.

Analyze the transcript from start to finish and identify distinct topics of discussion as they naturally emerge.

Detect topic boundaries by:

Changes in subject matter

New questions or prompts from the host

Shifts in technical, economic, social, or strategic focus

Assign each discussion segment a concise, descriptive topic label (3–10 words).

Preserve the order in which topics appear in the episode.

Merge adjacent segments if they clearly belong to the same topic to avoid redundancy.

Avoid segments like promotions, introductions of guests, or any other part that is not relevant for the actual discussion.

Context

You will be provided:

title: Podcast episode title

description: Episode description

transcript: Full podcast transcript (may include filler, speaker labels, timestamps, or transcription noise)

Your task is to extract the general topics of discussion in the order they are discussed.

Constraints

Output only valid JSON.

Output must be a single array of topic strings.

No additional fields, metadata, explanations, or prose.

Topics must be:

General (not overly granular)

Specific (not overly vague like “Space” or “Technology”)

Do not invent topics not supported by the transcript.

Aim for:

6–14 topics for typical episodes

Fewer if the episode is short; merge if necessary for long episodes

Examples

Example Input

"title": "Episode #517: How Orbital Robotics Turns Space Junk into Infrastructure",

"description": "...",

"transcript": "..."


Example Output

[

"Introduction and episode overview",

"Guest background in robotics",

"Challenges of orbital robotics",

"Autonomy and space debris management",

"Long-term vision for in-space recycling",

"Lunar resources and future space infrastructure"

]

Episode data:

Title: {title}

Description: {description}

Transcript: {transcript}
"""