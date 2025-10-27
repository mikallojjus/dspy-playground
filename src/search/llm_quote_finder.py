"""
DSPy-based quote finder for extracting supporting quotes from transcript chunks.

Uses DSPy's ChainOfThought to find quotes that directly support claims,
with optimization capability via BootstrapFewShot.

Usage:
    from src.search.llm_quote_finder import QuoteFinder

    # Use optimized model (recommended for production)
    finder = QuoteFinder(model_path="models/quote_finder_v1.json")

    # Or use zero-shot baseline (for testing)
    finder = QuoteFinder()

    # Find quotes
    quotes = finder(
        claim="Bitcoin reached $69,000 in November 2021",
        transcript_chunks="[transcript text here]"
    )

    for quote in quotes:
        print(f"Quote: {quote['text']}")
        print(f"Reasoning: {quote['reasoning']}")
"""

from typing import List, Dict, Optional
from pathlib import Path
import dspy

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class QuoteFinderSignature(dspy.Signature):
    """
    You are tasked with finding direct, verbatim quotes from podcast transcripts that
    support specific factual claims. Your goal is to extract EXACT word-for-word excerpts
    that provide evidence, context, or elaboration for the claim.
    """

    claim: str = dspy.InputField(
        desc=(
            "A factual claim that needs supporting evidence. "
            "The claim should be verifiable and specific."
        )
    )

    transcript_chunks: str = dspy.InputField(
        desc=(
            "Transcript excerpts from the podcast episode (SOURCE CONTENT). "
            "These sections have been pre-selected as potentially relevant to the claim. "
            "All quotes MUST come verbatim from this text - do not use external knowledge."
        )
    )

    quotes: List[Dict[str, str]] = dspy.OutputField(
        desc=(
            "List of direct quotes from the transcript that support the claim.\n\n"

            "CRITICAL REQUIREMENTS:\n"
            "1. Quotes must be EXACT, word-for-word excerpts from the transcript above\n"
            "2. Do not create, modify, paraphrase, or invent quotes\n"
            "3. Each quote must directly support the claim (provide evidence, context, or elaboration)\n"
            "4. If no supporting quotes exist in the transcript, return an empty list\n\n"

            "WHAT MAKES A QUOTE SUPPORTIVE:\n"
            "- Provides factual evidence for the claim\n"
            "- Gives context or background that validates the claim\n"
            "- Elaborates on or confirms details mentioned in the claim\n"
            "- Directly relates to the specific facts in the claim (not just the topic)\n\n"

            "OUTPUT FORMAT:\n"
            "[{\"text\": \"exact verbatim quote\", \"reasoning\": \"how this supports the claim\"}]\n\n"

            "DO NOT include quotes that are merely topically related but don't support the claim.\n"
            "DO NOT modify quotes to make them fit better - extract them EXACTLY as written."
        )
    )


class QuoteFinder(dspy.Module):
    """
    DSPy module for finding supporting quotes using Chain of Thought reasoning.

    This module can be optimized using DSPy's teleprompters (e.g., BootstrapFewShot)
    to learn better prompts and few-shot examples.

    Example:
        ```python
        import dspy

        # Configure DSPy
        lm = dspy.LM('ollama/qwen2.5:7b-instruct-q4_0', api_base='http://localhost:11434')
        dspy.configure(lm=lm)

        # Create finder
        finder = QuoteFinder()

        # Find quotes
        result = finder(
            claim="Bitcoin reached $69,000",
            transcript_chunks="In November 2021, Bitcoin hit $69k..."
        )

        print(f"Found {len(result.quotes)} quotes")
        for quote in result.quotes:
            print(f"- {quote['text']}")
        ```

    Optimization example:
        ```python
        from src.optimization.quote_finder_optimizer import QuoteFinderOptimizer

        optimizer = QuoteFinderOptimizer(lm, reranker, entailment_validator, training_data)
        optimized_finder = await optimizer.optimize()

        # Use optimized finder (has learned better prompts and examples)
        result = optimized_finder(claim=claim, transcript_chunks=chunks)
        ```
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the quote finder module.

        Args:
            model_path: Optional path to optimized DSPy model file.
                       If provided and exists, loads the optimized model with few-shot examples.
                       If None or file doesn't exist, uses zero-shot baseline.
                       Default: "models/quote_finder_v1.json"

        Example:
            ```python
            # Use optimized model
            finder = QuoteFinder(model_path="models/quote_finder_v1.json")

            # Use zero-shot baseline
            finder = QuoteFinder()
            ```
        """
        super().__init__()

        # Configure DSPy with Ollama (safe to call multiple times - uses global state)
        lm = dspy.LM(
            f"ollama/{settings.ollama_model}",
            api_base=settings.ollama_url,
            num_ctx=settings.ollama_num_ctx
        )
        dspy.configure(lm=lm)
        logger.debug(f"Configured DSPy LM with context size: {settings.ollama_num_ctx} tokens")

        # ChainOfThought: LLM will reason about which quotes support the claim
        self.find_quotes = dspy.ChainOfThought(QuoteFinderSignature)

        # Load optimized model if path provided
        if model_path:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                logger.info(f"Loading optimized QuoteFinder model from {model_path}")
                self.find_quotes.load(str(model_path_obj))

                # Log few-shot examples count
                if hasattr(self.find_quotes, 'demos') and self.find_quotes.demos:
                    demos = getattr(self.find_quotes, 'demos', [])
                    logger.info(f"Loaded model with {len(demos)} few-shot examples")
                else:
                    logger.info("Loaded model (zero-shot)")
            else:
                logger.warning(
                    f"Optimized model file not found at {model_path}, "
                    "using zero-shot baseline"
                )
        else:
            logger.info("Initialized QuoteFinder (zero-shot baseline)")

    def forward(self, claim: str, transcript_chunks: str) -> dspy.Prediction:
        """
        Find supporting quotes for a claim from transcript chunks.

        Args:
            claim: The factual claim to find support for
            transcript_chunks: Relevant transcript text (pre-filtered by semantic search)

        Returns:
            dspy.Prediction containing:
                - quotes: List[Dict[str, str]] with 'text' and 'reasoning' keys
                - rationale: Chain of thought reasoning (if available)

        Example:
            ```python
            result = finder(
                claim="Ethereum supports smart contracts",
                transcript_chunks="Speaker 1: Ethereum enables developers to build..."
            )

            for quote in result.quotes:
                print(f"Quote: {quote['text']}")
                print(f"Why it supports: {quote['reasoning']}")
            ```
        """
        logger.debug(f"Finding quotes for claim: {claim[:60]}...")

        # Call DSPy's ChainOfThought
        result = self.find_quotes(
            claim=claim,
            transcript_chunks=transcript_chunks
        )

        logger.debug(
            f"LLM returned {len(result.quotes) if isinstance(result.quotes, list) else 0} quotes"
        )

        return result


class QuoteFinderWithVerification(dspy.Module):
    """
    Extended quote finder that includes verification step.

    This module combines the base QuoteFinder with automatic verification
    to catch hallucinations before returning quotes.

    Example:
        ```python
        from src.search.quote_verification import QuoteVerifier

        verifier = QuoteVerifier()
        finder = QuoteFinderWithVerification(verifier=verifier)

        result = finder(
            claim="Bitcoin reached $69,000",
            transcript_chunks="..."
        )

        # result.quotes are guaranteed to be verified (hallucinations filtered)
        # result.verification_stats contains verification metrics
        ```
    """

    def __init__(self, verifier=None):
        """
        Initialize quote finder with verification.

        Args:
            verifier: QuoteVerifier instance (default: creates new one)
        """
        super().__init__()

        from src.search.quote_verification import QuoteVerifier

        self.find_quotes = dspy.ChainOfThought(QuoteFinderSignature)
        self.verifier = verifier or QuoteVerifier()

        logger.info("Initialized QuoteFinderWithVerification")

    def forward(self, claim: str, transcript_chunks: str) -> dspy.Prediction:
        """
        Find and verify quotes.

        Args:
            claim: The factual claim
            transcript_chunks: Relevant transcript text

        Returns:
            dspy.Prediction with verified quotes and verification stats
        """
        # Step 1: LLM finds candidate quotes
        result = self.find_quotes(
            claim=claim,
            transcript_chunks=transcript_chunks
        )

        candidate_quotes = result.quotes if isinstance(result.quotes, list) else []

        logger.debug(f"LLM returned {len(candidate_quotes)} candidate quotes")

        # Step 2: Verify each quote
        verified_quotes = []
        verification_stats = {
            "total_candidates": len(candidate_quotes),
            "verified": 0,
            "hallucinations": 0,
            "avg_confidence": 0.0
        }

        total_confidence = 0.0

        for quote_data in candidate_quotes:
            if not isinstance(quote_data, dict):
                logger.warning(f"Invalid quote format: {quote_data}")
                continue

            quote_text = quote_data.get("text", "")
            if not quote_text:
                continue

            # Verify quote exists in transcript
            verification_result = self.verifier.verify(
                quote_text,
                transcript_chunks,
                claim
            )

            if verification_result.is_valid:
                # Use corrected text from transcript (not LLM output)
                verified_quotes.append({
                    "text": verification_result.corrected_text,
                    "reasoning": quote_data.get("reasoning", ""),
                    "verification_confidence": verification_result.confidence,
                    "match_type": verification_result.match_type
                })
                verification_stats["verified"] += 1
                total_confidence += verification_result.confidence
            else:
                verification_stats["hallucinations"] += 1
                logger.warning(
                    f"Rejected hallucinated quote: {quote_text[:60]}... "
                    f"(claim: {claim[:60]}...)"
                )

        # Calculate average confidence
        if verification_stats["verified"] > 0:
            verification_stats["avg_confidence"] = (
                total_confidence / verification_stats["verified"]
            )

        logger.info(
            f"Verification complete: {verification_stats['verified']} verified, "
            f"{verification_stats['hallucinations']} rejected "
            f"(hallucination rate: {verification_stats['hallucinations'] / max(1, verification_stats['total_candidates']):.1%})"
        )

        # Return prediction with verified quotes
        return dspy.Prediction(
            quotes=verified_quotes,
            verification_stats=verification_stats,
            rationale=getattr(result, 'rationale', None)
        )
