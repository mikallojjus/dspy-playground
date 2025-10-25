"""
Fix pgvector schema migration.

This script:
1. Enables the pgvector extension
2. Alters the embedding column from double precision[] to vector(768)

Run this once to fix the schema mismatch between the database and SQLAlchemy models.

Usage:
    uv run python scripts/fix_pgvector_schema.py
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.database.connection import get_db_session_context
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def fix_pgvector_schema():
    """
    Fix the database schema to use pgvector types.
    """
    logger.info("Starting pgvector schema migration...")

    with get_db_session_context() as session:
        # 1. Enable pgvector extension
        logger.info("Enabling pgvector extension...")
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        session.commit()
        logger.info("SUCCESS: pgvector extension enabled")

        # 2. Alter the embedding column type
        logger.info("Altering crypto.claims.embedding column type...")
        session.execute(text("""
            ALTER TABLE crypto.claims
            ALTER COLUMN embedding TYPE vector(768)
            USING embedding::vector(768);
        """))
        session.commit()
        logger.info("SUCCESS: embedding column altered to vector(768)")

        # 3. Verify the change
        logger.info("Verifying column type...")
        result = session.execute(text("""
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'crypto'
            AND table_name = 'claims'
            AND column_name = 'embedding';
        """))

        for row in result:
            logger.info(f"Column info: {dict(row._mapping)}")

        logger.info("SUCCESS: pgvector schema migration completed successfully!")


if __name__ == "__main__":
    try:
        fix_pgvector_schema()
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        raise
