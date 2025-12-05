"""
Tag query service for CLI processing.

Provides helpers to fetch tags or filter them by creation date.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from src.database.connection import get_db_session
from src.database.models import Tag
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class TagQueryService:
    """
    Service for querying tags.

    Features:
    - Fetch all tags
    - Fetch tags in a created_at window
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize the query service.

        Args:
            db_session: Optional database session. If not provided, creates new session.
        """
        self.session = db_session or get_db_session()
        self._owns_session = db_session is None

    def __del__(self):
        """Close session if we own it."""
        if self._owns_session and hasattr(self, "session"):
            self.session.close()

    def get_all_tags(self) -> List[Tag]:
        """
        Get all tags ordered by creation date.

        Returns:
            List of Tag objects
        """
        logger.info("Fetching all tags")

        return (
            self.session.query(Tag)
            .order_by(Tag.created_at.desc().nulls_last(), Tag.id.desc())
            .all()
        )

    def get_tags_created_between(self, start: datetime, end: datetime) -> List[Tag]:
        """
        Get tags created between two datetimes (inclusive).

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of Tag objects within the range
        """
        if start > end:
            raise ValueError("start datetime must be before end datetime")

        logger.info(f"Fetching tags created between {start} and {end}")

        return (
            self.session.query(Tag)
            .filter(Tag.created_at >= start, Tag.created_at <= end)
            .order_by(Tag.created_at.desc().nulls_last(), Tag.id.desc())
            .all()
        )
