"""
ClaimRepository for database persistence operations.

Handles CRUD operations for claims, quotes, and claim-quote relationships.

Usage:
    from src.database.claim_repository import ClaimRepository

    repo = ClaimRepository(db_session)
    claim_ids = await repo.save_claims(claims_with_quotes, episode_id)
"""

from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.database.models import Claim, Quote, ClaimQuote
from src.extraction.quote_finder import ClaimWithQuotes, Quote as ExtractedQuote
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ClaimRepository:
    """
    Repository for database persistence of claims and quotes.

    Features:
    - Save claims with embeddings
    - Save unique quotes (reuses existing by position)
    - Create claim-quote junction records
    - Transaction management with rollback
    - Merge quotes to existing claims (for cross-episode dedup)

    Example:
        ```python
        repo = ClaimRepository(db_session)

        # Save new claims
        claim_ids = await repo.save_claims(claims_with_quotes, episode_id)

        # Merge quotes to existing claim (cross-episode dedup)
        await repo.merge_quotes_to_existing_claim(
            existing_claim_id=123,
            new_quotes=quote_list,
            episode_id=456
        )

        # Commit transaction
        db_session.commit()
        ```
    """

    def __init__(self, db_session: Session):
        """
        Initialize the repository.

        Args:
            db_session: SQLAlchemy database session

        Example:
            ```python
            from src.database.connection import get_db_session

            session = get_db_session()
            repo = ClaimRepository(session)
            ```
        """
        self.session = db_session

    async def save_claims(
        self,
        claims_with_quotes: List[ClaimWithQuotes],
        episode_id: int
    ) -> List[int]:
        """
        Save claims with their quotes to database.

        Creates:
        - Claim records with embeddings
        - Quote records (reusing existing when possible)
        - ClaimQuote junction records with relevance scores

        Args:
            claims_with_quotes: List of claims with supporting quotes
            episode_id: Episode ID these claims belong to

        Returns:
            List of claim IDs that were saved

        Example:
            ```python
            repo = ClaimRepository(session)
            claim_ids = await repo.save_claims(claims_with_quotes, episode_id=123)
            print(f"Saved {len(claim_ids)} claims")
            ```
        """
        if not claims_with_quotes:
            logger.warning("No claims to save")
            return []

        logger.info(f"Saving {len(claims_with_quotes)} claims for episode {episode_id}")

        claim_ids = []

        try:
            for i, claim_with_quotes in enumerate(claims_with_quotes, 1):
                # Save claim
                claim = Claim(
                    episode_id=episode_id,
                    claim_text=claim_with_quotes.claim_text,
                    confidence=claim_with_quotes.confidence,
                    embedding=None,  # Will be set after we get the ID
                    confidence_components=(
                        claim_with_quotes.confidence_components.__dict__
                        if claim_with_quotes.confidence_components else None
                    )
                )

                self.session.add(claim)
                self.session.flush()  # Get ID without committing

                claim_ids.append(claim.id)

                logger.debug(
                    f"Saved claim {i}/{len(claims_with_quotes)}: ID={claim.id}, "
                    f"text='{claim.claim_text[:60]}...'"
                )

                # Save quotes and create links
                if claim_with_quotes.quotes:
                    await self._save_quotes_and_links(
                        claim.id,
                        claim_with_quotes.quotes,
                        episode_id
                    )

            logger.info(f"✅ Saved {len(claim_ids)} claims")
            return claim_ids

        except Exception as e:
            logger.error(f"Error saving claims: {e}", exc_info=True)
            self.session.rollback()
            raise

    async def _save_quotes_and_links(
        self,
        claim_id: int,
        quotes: List[ExtractedQuote],
        episode_id: int
    ) -> None:
        """
        Save quotes and create claim-quote links.

        Reuses existing quotes when possible (by position).

        Args:
            claim_id: Claim ID to link quotes to
            quotes: List of extracted quotes
            episode_id: Episode ID

        Internal method called by save_claims.
        """
        for quote_obj in quotes:
            # Check if quote already exists (by position)
            existing_quote = (
                self.session.query(Quote)
                .filter(
                    Quote.episode_id == episode_id,
                    Quote.start_position == quote_obj.start_position,
                    Quote.end_position == quote_obj.end_position
                )
                .first()
            )

            if existing_quote:
                quote_id = existing_quote.id
                logger.debug(
                    f"Reusing existing quote ID={quote_id} "
                    f"(pos: {quote_obj.start_position}-{quote_obj.end_position})"
                )
            else:
                # Create new quote
                quote = Quote(
                    episode_id=episode_id,
                    quote_text=quote_obj.quote_text,
                    start_position=quote_obj.start_position,
                    end_position=quote_obj.end_position,
                    speaker=quote_obj.speaker,
                    timestamp_seconds=quote_obj.timestamp_seconds
                )

                self.session.add(quote)
                self.session.flush()

                quote_id = quote.id
                logger.debug(f"Created new quote ID={quote_id}")

            # Create claim-quote link
            claim_quote = ClaimQuote(
                claim_id=claim_id,
                quote_id=quote_id,
                relevance_score=quote_obj.relevance_score,
                match_confidence=quote_obj.relevance_score,  # Same for now
                match_type="reranked",
                entailment_score=quote_obj.entailment_score,
                entailment_relationship=quote_obj.entailment_relationship
            )

            self.session.add(claim_quote)

    async def merge_quotes_to_existing_claim(
        self,
        existing_claim_id: int,
        new_quotes: List[ExtractedQuote],
        episode_id: int
    ) -> int:
        """
        Add new quotes to an existing claim (cross-episode deduplication).

        Used when we find a duplicate claim in another episode and want to
        add the new episode's quotes to the existing claim.

        Args:
            existing_claim_id: ID of existing claim to add quotes to
            new_quotes: List of new quotes to add
            episode_id: Episode ID where new quotes come from

        Returns:
            Number of new quote links created

        Example:
            ```python
            # Found duplicate claim 123 from episode 456
            num_added = await repo.merge_quotes_to_existing_claim(
                existing_claim_id=123,
                new_quotes=new_episode_quotes,
                episode_id=456
            )
            print(f"Added {num_added} quotes to existing claim")
            ```
        """
        logger.info(
            f"Merging {len(new_quotes)} quotes to existing claim {existing_claim_id}"
        )

        added_count = 0

        try:
            for quote_obj in new_quotes:
                # Check if quote already exists
                existing_quote = (
                    self.session.query(Quote)
                    .filter(
                        Quote.episode_id == episode_id,
                        Quote.start_position == quote_obj.start_position,
                        Quote.end_position == quote_obj.end_position
                    )
                    .first()
                )

                if existing_quote:
                    quote_id = existing_quote.id
                else:
                    # Create new quote
                    quote = Quote(
                        episode_id=episode_id,
                        quote_text=quote_obj.quote_text,
                        start_position=quote_obj.start_position,
                        end_position=quote_obj.end_position,
                        speaker=quote_obj.speaker,
                        timestamp_seconds=quote_obj.timestamp_seconds
                    )

                    self.session.add(quote)
                    self.session.flush()
                    quote_id = quote.id

                # Check if link already exists
                existing_link = (
                    self.session.query(ClaimQuote)
                    .filter(
                        ClaimQuote.claim_id == existing_claim_id,
                        ClaimQuote.quote_id == quote_id
                    )
                    .first()
                )

                if existing_link:
                    logger.debug(
                        f"Link already exists for claim {existing_claim_id} "
                        f"and quote {quote_id}"
                    )
                    continue

                # Create new link
                claim_quote = ClaimQuote(
                    claim_id=existing_claim_id,
                    quote_id=quote_id,
                    relevance_score=quote_obj.relevance_score,
                    match_confidence=quote_obj.relevance_score,
                    match_type="reranked_crossepisode",
                    entailment_score=quote_obj.entailment_score,
                    entailment_relationship=quote_obj.entailment_relationship
                )

                self.session.add(claim_quote)
                added_count += 1

            self.session.flush()

            logger.info(
                f"✅ Added {added_count} new quote links to claim {existing_claim_id}"
            )

            return added_count

        except Exception as e:
            logger.error(f"Error merging quotes: {e}", exc_info=True)
            self.session.rollback()
            raise

    async def update_claim_embeddings(
        self,
        claim_ids_and_embeddings: Dict[int, List[float]]
    ) -> None:
        """
        Update embeddings for claims after initial insert.

        Embeddings are updated separately because they're generated after
        claim IDs are assigned.

        Args:
            claim_ids_and_embeddings: Dict mapping claim_id to embedding vector

        Example:
            ```python
            embeddings = {
                123: [0.1, 0.2, ...],  # 768 dims
                124: [0.3, 0.4, ...],
            }
            await repo.update_claim_embeddings(embeddings)
            ```
        """
        logger.info(f"Updating embeddings for {len(claim_ids_and_embeddings)} claims")

        try:
            for claim_id, embedding in claim_ids_and_embeddings.items():
                claim = self.session.query(Claim).get(claim_id)
                if claim:
                    claim.embedding = embedding
                else:
                    logger.warning(f"Claim {claim_id} not found for embedding update")

            self.session.flush()
            logger.info("✅ Embeddings updated")

        except Exception as e:
            logger.error(f"Error updating embeddings: {e}", exc_info=True)
            self.session.rollback()
            raise

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_claims(claims, episode_id)
                session.commit()
            except Exception as e:
                repo.rollback()
                raise
            ```
        """
        logger.warning("Rolling back transaction")
        self.session.rollback()

    def commit(self) -> None:
        """
        Commit current transaction.

        Example:
            ```python
            await repo.save_claims(claims, episode_id)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
