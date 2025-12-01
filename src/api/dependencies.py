"""FastAPI dependencies for dependency injection."""

from typing import Generator
from sqlalchemy.orm import Session
from src.database.connection import get_db_session


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting a database session.

    Yields a SQLAlchemy session and ensures it's closed after use.
    """
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()
