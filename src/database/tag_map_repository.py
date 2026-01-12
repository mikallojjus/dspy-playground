"""
TagMapRepository for database persistence of tag map records.

Handles CRUD operations for tag map relationships.

Usage:
    from src.database.tag_map_repository import TagMapRepository

    repo = TagMapRepository(db_session)
    updated_claims = await repo.save_tag_maps(claim_topics)
"""

from typing import Dict, List, Tuple
from sqlalchemy.orm import Session

from src.database.models import TagMap
from src.extraction.quote_finder import ClaimWithTopic
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class TagMapRepository:
    """
    Repository for database persistence of tag map records.

    Features:
    - Save tag map entries in batch
    - Query tag categories by claim-episode
    - Query claim-episode IDs by tag categories
    - Transaction management with rollback
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
            repo = TagMapRepository(session)
            ```
        """
        self.session = db_session

    async def save_tag_maps(
        self,
        claim_topics: List[ClaimWithTopic]
    ) -> List[ClaimWithTopic]:
        """
        Save tag map entries to database.

        Creates TagMap records for each claim-episode + topic pair, skipping
        any entries that already exist. Returns the ClaimWithTopic items that
        have valid tag map links.

        Args:
            claim_topics: ClaimWithTopic items with claim_episode_id and topic

        Returns:
            List of ClaimWithTopic with tag map entries saved
        """
        if not claim_topics:
            logger.warning("No claim topics provided to tag map")
            return []

        logger.info(f"Saving tag maps for {len(claim_topics)} claim topics")

        try:
            pair_to_claim_topics: Dict[Tuple[int, str], List[ClaimWithTopic]] = {}
            ordered_pairs: List[Tuple[int, str]] = []

            for claim_topic in claim_topics:
                claim_episode_id = claim_topic.claim_episode_id
                if claim_episode_id is None:
                    logger.warning(
                        "Claim topic missing claim_episode_id; skipping tag map creation"
                    )
                    continue

                tag_category = claim_topic.topic
                if not tag_category:
                    logger.warning(
                        "Claim topic missing topic; skipping tag map creation"
                    )
                    continue

                key = (claim_episode_id, tag_category)
                if key not in pair_to_claim_topics:
                    pair_to_claim_topics[key] = []
                    ordered_pairs.append(key)

                pair_to_claim_topics[key].append(claim_topic)

            if not ordered_pairs:
                logger.warning("No claim episode IDs provided to tag map")
                return []

            claim_episode_ids = [
                claim_episode_id for claim_episode_id, _ in ordered_pairs
            ]
            existing_rows = (
                self.session.query(
                    TagMap.from_claim_episode_id,
                    TagMap.tag_category
                )
                .filter(TagMap.from_claim_episode_id.in_(claim_episode_ids))
                .all()
            )
            existing_pairs = {(row[0], row[1]) for row in existing_rows}

            new_pairs = [
                pair
                for pair in ordered_pairs
                if pair not in existing_pairs
            ]

            if new_pairs:
                tag_maps = [
                    TagMap(
                        from_claim_episode_id=claim_episode_id,
                        tag_category=tag_category
                    )
                    for claim_episode_id, tag_category in new_pairs
                ]
                self.session.add_all(tag_maps)
                self.session.flush()

            saved_claim_topics: List[ClaimWithTopic] = []
            for key, claim_topic_list in pair_to_claim_topics.items():
                if key in existing_pairs or key in new_pairs:
                    saved_claim_topics.extend(claim_topic_list)

            logger.info(
                "Saved %s new tag map entries (out of %s unique)",
                len(new_pairs),
                len(ordered_pairs)
            )

            return saved_claim_topics

        except Exception as e:
            logger.error(f"Error saving tag map entries: {e}", exc_info=True)
            self.session.rollback()
            raise

    def get_tag_categories_by_claim_episode(
        self,
        claim_episode_id: int
    ) -> List[str]:
        """
        Get all tag categories linked to a specific claim-episode record.

        Args:
            claim_episode_id: ClaimEpisode ID to fetch tags for

        Returns:
            List of tag category strings
        """
        logger.info(
            f"Fetching tag categories for claim_episode_id {claim_episode_id}"
        )

        tag_categories = [
            row[0]
            for row in (
                self.session.query(TagMap.tag_category)
                .filter(TagMap.from_claim_episode_id == claim_episode_id)
                .all()
            )
        ]

        logger.info(
            f"Found {len(tag_categories)} tag categories for claim_episode_id {claim_episode_id}"
        )

        return tag_categories

    def get_claim_episode_ids_by_tags(
        self,
        tag_categories: List[str]
    ) -> Dict[str, List[int]]:
        """
        Get claim-episode IDs for a list of tag categories.

        Args:
            tag_categories: List of tag categories to fetch claim-episode IDs for

        Returns:
            Dict mapping tag_category to list of claim-episode IDs
        """
        if not tag_categories:
            logger.warning("No tag categories provided to fetch claim episodes")
            return {}

        logger.info(
            f"Fetching claim-episode IDs for {len(tag_categories)} tag categories"
        )

        rows = (
            self.session.query(
                TagMap.tag_category,
                TagMap.from_claim_episode_id
            )
            .filter(TagMap.tag_category.in_(tag_categories))
            .all()
        )

        claim_episodes_by_tag = {tag_category: [] for tag_category in tag_categories}
        for tag_category, claim_episode_id in rows:
            claim_episodes_by_tag.setdefault(tag_category, []).append(
                claim_episode_id
            )

        logger.info(
            f"Found claim-episode links for {len(claim_episodes_by_tag)} tag categories"
        )

        return claim_episodes_by_tag

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_tag_maps(claim_topics)
                session.commit()
            except Exception:
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
            await repo.save_tag_maps(claim_topics)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
