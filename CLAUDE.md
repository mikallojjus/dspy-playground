# CLAUDE GUIDELINES

## Description

This is a repository where we do small scale experiments with DSPy

The reason for this experiment is to gain knowledge on how DSPy works and how it fits our usecase.

We are building a podcast app and we want to extract meaningful information from worlds podcasts.

Short summary of our LLM pipeline: we want to extract claims and quotes from the transcript. We want to extract meaningful claims mentioned in the podcast and provide quotes for them. Quote should contain a timestamp.

We use qwen 2.5 7b llm model running on Ollama.
We have 2 different calls to LLM - claim extraction and entailment. Entailment is determining the quotes relation to the claim - we only want quotes that directly support the claim, not just relate to it.

Quotes were being found using:

```
  In TranscriptSearchIndex.findQuotesForClaim() (line 42), quotes are found using semantic similarity:
  - Embeds the claim text
  - Compares with all transcript segments using cosine similarity
```

We want to optimize each prompt

Things to consider:

- Context window and transcript chunking
- Transcript segmentation - we want to

## DSPy Optimization Goals

**Primary Goal:** Optimize 2 prompts with measurable improvement

1. **Claim Extraction Prompt** - Improve claim quality (reduce vague/opinion claims)
    - Baseline: 40% of extracted claims are low-quality
    - Target: <15% low-quality claims

2. **Entailment Validation Prompt** - Reduce false positives (RELATED misclassified as SUPPORTS)
    - Baseline: 30% false positive rate
    - Target: <10% false positive rate

## Quality Criteria

**Good Claim:** Factual, self-contained, no pronouns, specific, verifiable
**Bad Claim:** Opinion, vague, requires context, uses "he/she/they"

**Good Entailment (SUPPORTS):** Quote directly asserts or provides clear evidence
**Bad Entailment (misclassified RELATED):** Topically related but doesn't validate claim

## Sample data

### Transcript example

```
0 (0s):
I'm only trusting the L two to provide me the nice stuff. I'm not trusting it to provide me the core stuff. The core Property, Rights and censorship resistance that Ethereum is gonna gimme.

2 (16s):
Welcome to Bankless where today we explore a defense of the Ethereum roadmap. This is Ryan Sean Adams. I'm here with David Hoffman and we are here to help you become more bankless. The Ethereum roadmap has been called into question recently, I think Bankless has aired some of these dissents, and I'll say maybe two things about that before we begin. First is this. I think dissenting opinions are really important for us to consider and David and I will continue voicing them on Bankless and even when they're wrong, I think at best they help sharpen our ideas and both David and I would rather err on the side of open engagement than live in an echo chamber. So that's the first thing. The second I just think we have to recognize, yes, all this roadmap angst recently could totally just be a price thing.

2 (59s):
As we know in crypto, narrative follows price and when price is down, people find all sorts of reasons to doubt their conviction. And on this time will tell. But today's episode voices a counter to the doubters on the Ethereum roadmap. Mike Neuder is an Ethereum developer and a previous Bankless guest. He thinks the Ethereum roadmap is pretty great as is it's right on track. in fact, his message is that we should stay the course, play the long game and watch the value of ETH grow as a monetary asset, as the Ethereum economy grows across all sorts of layer twos.
```

### Prompt that was initially used for claim extraction

```
You are analyzing a podcast transcript for factual data.

Create verifiable factual claims using the data from the transcript below.

Include all factual statements that are explicitly mentioned in the transcript, regardless of their relevance to any specific individual, company or entity unless otherwise specified.
You should not try to limit the number of created claims.



A claim is any statement that can be fact-checked or verified, such as:

- Statistics or numbers (e.g., "Tesla's revenue grew 40%")
- Product features (e.g., "The new iPhone has a titanium frame")
- Historical events (e.g., "Bitcoin was created in 2009")
- Company announcements (e.g., "Microsoft acquired Activision for $68 billion")
- Scientific facts (e.g., "Water boils at 100 degrees Celsius")

Return a JSON object with the following structure:

{

"claims": [

"The concise factual claim”
“Quote that supports the claim”,
"Another factual claim"
“Quote that supports the claim”,

"..."

]

}

Rules:

1. Create only factual claims that can be verified using the given transcript

2. Keep claims concise and specific (one claim per statement)

3. DO NOT include opinions, questions, or speculative statements
4. DO NOT hallucinate or infer claims not explicitly stated
5. If no verifiable claims are found, return {"claims": []}
6. Focus on extracting accurate claim text - supporting evidence will be found separately
7.Do not use pronouns (e.g., “he,” “she,” “they”) or vague titles (e.g., “the CEO,” “the founder”) while creating or rephrasing the claim.
8.If the person’s full name is not explicitly mentioned in the WHOLE transcript, skip the claim entirely.
8.ALL created claim must be rephrased at least once.
9.Make sure that you included ALL important data from the transcript.



Example:

Input: "Apple announced the iPhone 15 yesterday with a new titanium design. Tim Cook said it's 20% lighter than the previous model. I think it looks great!"

Output: {

"claims": [

"iPhone 15 has a titanium design",

"iPhone 15 is 20% lighter than the previous model"

]

}

Note: "I think it looks great" is an opinion, not a factual claim, so it's excluded.

Transcript:
[TRANSCRIPT CHUNK IS INSERTED HERE]
```

### Prompt initially used for entailment

```
Analyze whether this quote provides evidence for or supports the claim.

CLAIM: "${claim}"
QUOTE: "${quote}"

Does the quote directly assert the claim or provide clear evidence that supports it?

Respond ONLY with a valid JSON object (no markdown, no code blocks):
{
  "relationship": "SUPPORTS|RELATED|NEUTRAL|CONTRADICTS",
  "reasoning": "brief explanation in one sentence",
  "confidence": 0.0-1.0
}

Guidelines:
- SUPPORTS: Quote directly asserts the claim or provides clear evidence that validates it
- RELATED: Quote is topically related but doesn't validate or provide evidence for the claim
- NEUTRAL: Quote is unrelated or provides no evidence
- CONTRADICTS: Quote contradicts or undermines the claim

Be strict: only mark as SUPPORTS if the quote genuinely provides evidence.
```

### DB Structure

```
CREATE TABLE IF NOT EXISTS crypto.claims
(
    id bigint NOT NULL DEFAULT nextval('crypto.claims_id_seq'::regclass),
    episode_id bigint NOT NULL,
    claim_text text COLLATE pg_catalog."default" NOT NULL,
    confidence double precision NOT NULL,
    metadata jsonb,
    embedding double precision[],
    confidence_components jsonb,
    reranker_scores jsonb,
    CONSTRAINT "PK_96c91970c0dcb2f69fdccd0a698" PRIMARY KEY (id),
    CONSTRAINT fk_claims_episode FOREIGN KEY (episode_id)
        REFERENCES crypto.podcast_episodes (id) MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE
)

CREATE TABLE IF NOT EXISTS crypto.quotes
(
    id bigint NOT NULL DEFAULT nextval('crypto.quotes_id_seq'::regclass),
    episode_id bigint NOT NULL,
    quote_text text COLLATE pg_catalog."default" NOT NULL,
    start_position integer,
    end_position integer,
    speaker character varying(255) COLLATE pg_catalog."default",
    metadata jsonb,
    created_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone NOT NULL DEFAULT CURRENT_TIMESTAMP,
    timestamp_seconds integer,
    CONSTRAINT "PK_99a0e8bcbcd8719d3a41f23c263" PRIMARY KEY (id),
    CONSTRAINT "FK_60a8d735219758f176538318322" FOREIGN KEY (episode_id)
        REFERENCES crypto.podcast_episodes (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

CREATE TABLE IF NOT EXISTS crypto.claim_quotes
(
    claim_id bigint NOT NULL,
    quote_id bigint NOT NULL,
    relevance_score double precision NOT NULL,
    match_confidence double precision NOT NULL,
    match_type character varying(20) COLLATE pg_catalog."default",
    metadata jsonb,
    entailment_score double precision,
    entailment_relationship character varying(20) COLLATE pg_catalog."default",
    CONSTRAINT "PK_cf45d74a14d0f0e71bfe7cbbf2a" PRIMARY KEY (claim_id, quote_id),
    CONSTRAINT "FK_ad965e311c553a3c34bb3b83ff1" FOREIGN KEY (quote_id)
        REFERENCES crypto.quotes (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE,
    CONSTRAINT "FK_b3d3d89330d96ecd94675a1db8c" FOREIGN KEY (claim_id)
        REFERENCES crypto.claims (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

CREATE TABLE IF NOT EXISTS crypto.podcast_episodes
(
    id bigint NOT NULL DEFAULT nextval('crypto.podcast_episodes_id_seq'::regclass),
    podcast_id bigint NOT NULL,
    name text COLLATE pg_catalog."default" NOT NULL,
    description text COLLATE pg_catalog."default",
    episode_number integer,
    duration integer,
    published_at date,
    logo text COLLATE pg_catalog."default",
    season text COLLATE pg_catalog."default",
    episode_type text COLLATE pg_catalog."default",
    audio_url text COLLATE pg_catalog."default",
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    transcript text COLLATE pg_catalog."default",
    CONSTRAINT podcast_episodes_pkey PRIMARY KEY (id),
    CONSTRAINT fk_podcast_episodes_podcast FOREIGN KEY (podcast_id)
        REFERENCES crypto.podcasts (id) MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE
)
```

## Descriptors

```
# Claim

## Descriptors that explain what the claim is:

- A claim is an assertion of a specific fact, explanation or viewpoint presented in a form of a text statement
- Claim is self-contained and must carry a clear and complete meaning without requiring any additional context or other claims.
- A claim is representing a certain possible state of reality, it does not have an author

## Descriptors that explain how the claim is structured:

- The claim should avoid pronouns to ensure context independence and self-sufficiency; if any entity is referenced, it should be named and spelled explicitly.
- The tone should be neutral and professional, avoiding emotionally charged language.
- The tone and style should focus on word efficiency and clarity, avoiding unnecessary long sentences but also making sure that information and meaning is fully conveyed.
- Each claim should consist of one or two sentences.

## Descriptors that explain how the claim relates to other types:

- Claims are organized as any other types by assigning related topics
- Claims are supported by relevant quotes in the “Quotes that support the claim” property. Quotes provide context, which restates or elaborates on the accuracy of the claim or the fact that the claim was indeed expressed in the sources
- Each claim usually has multiple supporting quotes that come from multiple different sources.

# Quote

# Descriptors that explain what the type is:

- A quote is a meaningful and direct, word-for-word excerpt from a text, speech, document, dialogue, or other source.
- Quotes must always accurately represent what someone has said or written, in their exact form without any alterations; they must always be verbatim.
- Quotes have contextual information such as who said/wrote the quote, when it was said/written, where it was said/written. This contextual information is not a part of the quote itself.

# Descriptors that explain how the type is used:

- Quotes are used to support claims.
- Quotes provide context, reinforce, or elaborate on the points or accuracy of claims.
- Quotes must be used strategically to provide validity, credibility, or proof of the claim that it supports.
- Each quote should be carefully selected to match and support the key proposition of the claim or to prove that the claim is valid.
- The length of the quote must be carefully tailored so that enough context would be included in the quote to properly support the claim.
- Quotes can support claims in multiple ways: 1. Quote can be of an authoritative person/institution which expresses the same point or meaning as the claim. 2. Quotes can serve as evidence for the points made in a claim.
- The reader uses the quote to understand the context of the claim, thus proving or elaborating on the claim’s validity.

# Descriptors that explain the source of the type:

- Quotes come from official sources or relevant, involved parties who provide authoritative perspectives or confirmations.
- The usual sources (authors/publishers) of the quotes are parties directly involved in the matter (people, companies, institutions, etc.), news publishers that write articles about the matter, companies, or institutions that said or published something that’s relevant to the matter.
- Quotes usually come from news articles, official statements, official websites, documents, research papers.

# Descriptors that explain how the type relates to other types:

- A single claim usually has multiple supporting quotes from different sources.
- Quotes must directly support the claims or points made in the claims.
```

## Old architecture

We previously have built this pipeline in another repository, and it's documentation is provided in `specs/OLD_ARCHITECTURE.md`. The problem with that implementation was that we were hardcoding prompts and tuning them in isolation and we were flying blind on how prompt changes affect the results. This is where DSPy could be useful.

We do not have to replicate the old architecture immediately, but it should provide context of what we are trying to build.

### AI Summary of previous implementation

```
  Claims & Quotes Architecture Overview

  The system implements a two-pass extraction pipeline that processes podcast episode transcripts to identify factual claims and their supporting quotes. The core data model consists of three entities:
  Claim (factual statements with confidence scores and 768-dimensional embeddings), Quote (text excerpts with position tracking and speaker metadata), and ClaimQuote (many-to-many junction table storing
  relevance scores, entailment relationships, and validation metadata). Episodes are chunked with 1000-character overlaps to fit LLM context windows (~4096 tokens), then Pass 1 uses Ollama (Qwen2.5 7B) to
  extract raw claims from each chunk, while Pass 2 builds a semantic search index of the transcript to find and rank supporting quotes for each claim using cosine similarity followed by a mandatory
  reranker service that scores quote-claim relevance. Each quote undergoes LLM-based entailment validation to verify it actually supports the claim (SUPPORTS/RELATED/NEUTRAL/CONTRADICTS relationships),
  with quotes scoring below 0.7 entailment filtered out, resulting in typical 20-30% rejection rates.

  The system employs a sophisticated three-stage deduplication strategy operating at multiple levels. Within-episode deduplication uses embedding similarity (cosine ≥0.85) to group semantically identical
  claims from overlapping chunks, keeping the highest-confidence claim per group while merging their quotes. Cross-episode database deduplication uses pgvector to search all episodes for similar claims (L2
   distance <0.30, approximately 85% cosine similarity), then confirms duplicates via reranker verification (score ≥0.7) which catches paraphrased claims with different wording—this threshold was
  specifically tuned down from 0.9 because the stricter value missed semantic duplicates like "enables freedom" vs "gives freedom." When duplicates are found across episodes, the system either merges
  quotes into the existing claim (if new claim has higher confidence) or skips the new claim entirely (if existing has higher confidence). Finally, **quote-level global deduplication** detects duplicate
  quotes across all claims using position-based overlap (>50% overlap), text normalization, and token-based similarity (Jaccard ≥0.8), preserving the longest text and highest relevance score during merges.

  Confidence scoring combines multiple factors with weighted components: average quote relevance (60%), maximum quote relevance (20%), and quote count with diminishing returns capped at 5 quotes (20%),
  optionally incorporating average entailment scores when validation is enabled. The reranker service (running via Docker on localhost:8080) is absolutely critical—the system intentionally throws errors
  and halts if the service is unavailable rather than degrading gracefully, as accurate semantic similarity is required for both quote ranking and cross-episode deduplication. The entire pipeline is
  orchestrated by the ClaimExtractor service which coordinates OllamaClient (LLM extraction), EmbeddingService (nomic-embed-text for 768-dim vectors), RerankerService (semantic relevance scoring),
  QuoteValidator (entailment checking), and various deduplicators, ultimately persisting claims with their confidence components, embeddings, and deduplicated quote associations to PostgreSQL with pgvector
   support for efficient similarity searches.
```

## Rules

- Do not overengineer
- Deliver immediate and testable value, rather than planning for years in the future
- Call bullshit on bad technical decisions and call out developer when he is wrong
- Be disagreeable and opinionated
- Keep configuration minimal
- Remove obsolete code as you go
- Ensure no unused code is left
- We should spend time to evaluate our ideas, not blindly follow the rules
