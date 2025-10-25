"""
Database connection management using SQLAlchemy.

Provides session factory and connection pooling for PostgreSQL with pgvector support.

Usage:
    from src.database.connection import get_db_session

    session = get_db_session()
    try:
        episodes = session.query(PodcastEpisode).limit(5).all()
        # ... work with episodes ...
    finally:
        session.close()

    # Or use as context manager:
    with get_db_session() as session:
        episodes = session.query(PodcastEpisode).limit(5).all()
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.config.settings import settings
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False,  # Set to True for SQL query logging
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db_session() -> Session:
    """
    Get a new database session.

    Returns:
        SQLAlchemy Session instance

    Example:
        ```python
        session = get_db_session()
        try:
            episodes = session.query(PodcastEpisode).limit(5).all()
            print(f"Found {len(episodes)} episodes")
        finally:
            session.close()
        ```
    """
    return SessionLocal()


@contextmanager
def get_db_session_context() -> Generator[Session, None, None]:
    """
    Get a database session as a context manager.

    Automatically handles session cleanup.

    Yields:
        SQLAlchemy Session instance

    Example:
        ```python
        with get_db_session_context() as session:
            episodes = session.query(PodcastEpisode).limit(5).all()
            print(f"Found {len(episodes)} episodes")
        # Session is automatically closed
        ```
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful, False otherwise

    Example:
        ```python
        if test_connection():
            print("Database connection OK")
        else:
            print("Database connection FAILED")
        ```
    """
    try:
        with get_db_session_context() as session:
            # Simple query to test connection
            session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}", exc_info=True)
        return False
