"""
DSPy Claim Extractor Model.

Loads the optimized claim extraction model trained with LLM-as-judge metric.

Usage:
    from src.dspy_models.claim_extractor import ClaimExtractorModel

    extractor = ClaimExtractorModel()
    claims = extractor.extract_claims("Bitcoin reached $69,000 in November 2021.")
    print(claims)  # ["Bitcoin reached $69,000 in November 2021"]
"""

import dspy
from typing import List
from pathlib import Path

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def fix_split_claims(claims: List[str]) -> List[str]:
    """
    Fix claims that were incorrectly split by json_repair library.

    The json_repair library (used by DSPy 3.0.3) has a bug where it splits
    strings containing apostrophes when the LLM generates mixed quote styles.

    For example:
        Input:  ["Netanyahu", "s family faced persecution..."]
        Output: ["Netanyahu's family faced persecution..."]

    This function detects and merges such split claims.

    Args:
        claims: List of claim strings that may contain splits

    Returns:
        List of claims with splits merged back together
    """
    if not claims:
        return claims

    fixed = []
    i = 0

    while i < len(claims):
        claim = claims[i].strip()

        # Check if this looks like the first part of a split claim
        # Pattern: A proper noun or word that's not too long, followed by a claim starting with "s "
        if i + 1 < len(claims):
            next_claim = claims[i + 1].strip()

            # Check if next claim starts with "s " (indicating it's the tail of a possessive)
            if next_claim.startswith('s ') and len(claim) < 30:
                # This looks like a split possessive, merge them
                merged = claim + "'" + next_claim
                fixed.append(merged)
                logger.debug(f"Merged split claim: {repr(claim)} + {repr(next_claim)} -> {repr(merged)}")
                i += 2
                continue

        # No split detected, keep claim as-is
        fixed.append(claim)
        i += 1

    return fixed


class ClaimExtraction(dspy.Signature):
    """
    Extract factual, verifiable claims from podcast transcript text.

    Claims should be:
    - Factual (not opinions)
    - Self-contained (no pronouns like he/she/they without clear referents)
    - Specific (include names, numbers, dates when relevant)
    - Concise (5-40 words)

    OUTPUT FORMAT REQUIREMENT:
    Return claims as a valid JSON array using ONLY double quotes.
    Example: ["claim one", "claim two", "claim three"]
    Do NOT use single quotes or mix quote styles - use double quotes only.
    """

    transcript_chunk: str = dspy.InputField(desc="The podcast transcript text to analyze")
    claims: List[str] = dspy.OutputField(
        desc='List of factual claims as JSON array with double quotes only: ["claim1", "claim2"]'
    )


class ClaimExtractorModel:
    """
    DSPy-based claim extractor using optimized model.

    Loads the optimized model from models/claim_extractor_llm_judge_v1.json
    which was trained using BootstrapFewShot with LLM-as-judge metric.

    Attributes:
        model: DSPy ChainOfThought module with few-shot examples
    """

    def __init__(self, model_path: str = "models/claim_extractor_llm_judge_v1.json"):
        """
        Initialize the claim extractor model.

        Args:
            model_path: Path to the optimized DSPy model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If DSPy configuration fails
        """
        # NOTE: No monkey-patch needed with format="json" (Track A)
        # Ollama generates only valid JSON, preventing malformed output at source

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Optimized model not found at {model_path}. "
                "Run claim extraction optimization experiment first."
            )

        # Note: We use dspy.context() instead of dspy.configure() because:
        # 1. Global dspy.configure() happens at module import (src/config/dspy_config.py)
        # 2. This __init__ may run in async contexts during lazy initialization
        # 3. dspy.context() is safe for async tasks, dspy.configure() is not
        # 4. dspy.asyncify() captures the context-configured model for async execution

        logger.info(f"Configuring DSPy with Ollama at {settings.ollama_url}")

        # 🎯 STRUCTURED OUTPUT FIX: Use JSON schema for guided decoding
        # Research shows format="json" alone is insufficient (Ollama hallucinates structure)
        # JSON schema provides CONSTRAINED generation matching exact output structure
        #
        # WHY THIS WORKS IN INFERENCE (but not training):
        # - Inference only uses ClaimExtraction signature (reasoning, claims)
        # - Training uses multiple signatures (ClaimExtraction + ClaimQualityJudge)
        # - JSON schema must match signature, so we can only use it when one signature is used
        claims_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning about claim extraction"
                },
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of factual claims extracted from transcript"
                }
            },
            "required": ["reasoning", "claims"]
        }

        # Use dspy.context() to override global config with JSON schema
        # This is safe in async contexts (recommended by DSPy)
        from src.config.dspy_config import get_lm_with_schema
        custom_lm = get_lm_with_schema(claims_schema)

        with dspy.context(lm=custom_lm):
            logger.info("Configured with structured output (JSON schema for guided decoding)")

            # Load optimized model
            logger.info(f"Loading optimized claim extractor from {model_path}")
            self.model = dspy.ChainOfThought(ClaimExtraction)
            self.model.load(str(self.model_path))

        # Log few-shot examples count
        if hasattr(self.model, 'demos') and self.model.demos:
            logger.info(f"Loaded model with {len(self.model.demos)} few-shot examples")
        else:
            logger.info("Loaded model (zero-shot)")

        # Create async wrapper using dspy.asyncify for parallel processing
        # This captures the current DSPy configuration context
        self._async_model = dspy.asyncify(self.model)
        logger.debug("Created async wrapper for parallel claim extraction")

    def extract_claims(self, transcript_chunk: str) -> List[str]:
        """
        Extract claims from a transcript chunk (synchronous).

        Args:
            transcript_chunk: Text from podcast transcript

        Returns:
            List of extracted claim strings

        Example:
            ```python
            extractor = ClaimExtractorModel()
            claims = extractor.extract_claims(
                "Bitcoin reached $69,000 in November 2021. "
                "This was the all-time high for the cryptocurrency."
            )
            # Returns: [
            #     "Bitcoin reached $69,000 in November 2021",
            #     "Bitcoin's all-time high was $69,000"
            # ]
            ```
        """
        try:
            result = self.model(transcript_chunk=transcript_chunk)
            claims = getattr(result, 'claims', None) or []

            # Fix claims that were split by json_repair bug
            claims = fix_split_claims(claims)

            logger.debug(f"Extracted {len(claims)} claims from chunk ({len(transcript_chunk)} chars)")

            return claims

        except Exception as e:
            logger.error(f"Error extracting claims: {e}", exc_info=True)
            return []

    async def extract_claims_async(self, transcript_chunk: str) -> List[str]:
        """
        Extract claims from a transcript chunk (asynchronous using dspy.asyncify).

        This method uses dspy.asyncify to run the DSPy model in a worker thread
        while properly propagating the DSPy configuration context. This enables
        true parallel processing of multiple chunks.

        Args:
            transcript_chunk: Text from podcast transcript

        Returns:
            List of extracted claim strings

        Example:
            ```python
            extractor = ClaimExtractorModel()
            claims = await extractor.extract_claims_async(
                "Bitcoin reached $69,000 in November 2021."
            )
            # Returns: ["Bitcoin reached $69,000 in November 2021"]
            ```

        Note:
            This method is preferred over extract_claims() when processing
            multiple chunks in parallel, as it properly handles DSPy's
            thread-local configuration.
        """
        try:
            result = await self._async_model(transcript_chunk=transcript_chunk)
            claims = getattr(result, 'claims', None) or []

            # Fix claims that were split by json_repair bug
            claims = fix_split_claims(claims)

            logger.debug(f"Extracted {len(claims)} claims from chunk ({len(transcript_chunk)} chars)")

            return claims

        except Exception as e:
            logger.error(f"Error in async claim extraction: {e}", exc_info=True)
            return []

    def extract_claims_batch(self, transcript_chunks: List[str]) -> List[List[str]]:
        """
        Extract claims from multiple transcript chunks.

        Args:
            transcript_chunks: List of transcript text chunks

        Returns:
            List of claim lists (one per chunk)

        Example:
            ```python
            extractor = ClaimExtractorModel()
            results = extractor.extract_claims_batch([chunk1, chunk2, chunk3])
            # Returns: [
            #     ["claim1 from chunk1", "claim2 from chunk1"],
            #     ["claim1 from chunk2"],
            #     ["claim1 from chunk3", "claim2 from chunk3", "claim3 from chunk3"]
            # ]
            ```
        """
        logger.info(f"Extracting claims from {len(transcript_chunks)} chunks")

        results = []
        for i, chunk in enumerate(transcript_chunks, 1):
            claims = self.extract_claims(chunk)
            results.append(claims)
            logger.debug(f"Chunk {i}/{len(transcript_chunks)}: {len(claims)} claims")

        total_claims = sum(len(claims) for claims in results)
        logger.info(f"Extracted {total_claims} total claims from {len(transcript_chunks)} chunks")

        return results
