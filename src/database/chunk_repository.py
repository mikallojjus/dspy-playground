"""
ChunkRepository for database persistence of transcript chunks.

Handles CRUD operations for transcript chunks.

Usage:
    from src.database.chunk_repository import ChunkRepository

    repo = ChunkRepository(db_session)
    chunk_ids = await repo.save_chunks(chunks, episode_id)
"""

from typing import List
from sqlalchemy.orm import Session

from src.database.models import TranscriptChunk
from src.preprocessing.chunking_service import TextChunk
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ChunkRepository:
    """
    Repository for database persistence of transcript chunks.

    Features:
    - Save transcript chunks with position tracking
    - Query chunks by episode
    - Query chunks with associated claims
    - Transaction management with rollback

    Example:
        ```python
        repo = ChunkRepository(db_session)

        # Save chunks
        chunk_ids = await repo.save_chunks(chunks, episode_id)

        # Query chunks for episode
        chunks = await repo.get_chunks_by_episode(episode_id)

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
            repo = ChunkRepository(session)
            ```
        """
        self.session = db_session

    async def save_chunks(self, chunks: List[TextChunk], episode_id: int) -> List[int]:
        """
        Save transcript chunks to database.

        Creates TranscriptChunk records with position tracking.
        Returns the database IDs in the same order as the input chunks.

        Args:
            chunks: List of text chunks to save
            episode_id: Episode ID these chunks belong to

        Returns:
            List of chunk database IDs (in same order as input)

        Example:
            ```python
            repo = ChunkRepository(session)
            chunk_ids = await repo.save_chunks(chunks, episode_id=123)
            print(f"Saved {len(chunk_ids)} chunks")

            # Map chunk_id to database ID
            chunk_id_mapping = {
                chunk.chunk_id: db_id
                for chunk, db_id in zip(chunks, chunk_ids)
            }
            ```
        """
        if not chunks:
            logger.warning("No chunks to save")
            return []

        logger.info(f"Saving {len(chunks)} chunks for episode {episode_id}")

        chunk_ids = []

        try:
            for chunk in chunks:
                # Create chunk record
                db_chunk = TranscriptChunk(
                    episode_id=episode_id,
                    chunk_index=chunk.chunk_id,
                    chunk_text=chunk.text,
                    start_position=chunk.start_position,
                    end_position=chunk.end_position,
                )

                self.session.add(db_chunk)
                self.session.flush()  # Get ID without committing

                chunk_ids.append(db_chunk.id)

                logger.debug(
                    f"Saved chunk {chunk.chunk_id}: "
                    f"ID={db_chunk.id}, pos={chunk.start_position}-{chunk.end_position}, "
                    f"{len(chunk.text)} chars"
                )

            logger.info(f"✅ Saved {len(chunk_ids)} chunks")
            return chunk_ids

        except Exception as e:
            logger.error(f"Error saving chunks: {e}", exc_info=True)
            self.session.rollback()
            raise

    def rollback(self) -> None:
        """
        Rollback current transaction.

        Example:
            ```python
            try:
                await repo.save_chunks(chunks, episode_id)
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
            await repo.save_chunks(chunks, episode_id)
            repo.commit()
            ```
        """
        logger.info("Committing transaction")
        self.session.commit()
