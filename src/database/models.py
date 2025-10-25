"""
SQLAlchemy models for the claim-quote extraction pipeline.

Models match the PostgreSQL schema with pgvector support for embeddings.
All tables use the 'crypto' schema.

Models:
    - PodcastEpisode: Podcast episode with transcript
    - Claim: Extracted factual claim with embedding
    - Quote: Text excerpt from transcript
    - ClaimQuote: Many-to-many relationship between claims and quotes
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    DateTime,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class PodcastEpisode(Base):
    """
    Podcast episode with transcript.

    Table: crypto.podcast_episodes
    """

    __tablename__ = "podcast_episodes"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    podcast_id = Column(BigInteger, nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text)
    episode_number = Column(Integer)
    duration = Column(Integer)
    published_at = Column(Date)
    logo = Column(Text)
    season = Column(Text)
    episode_type = Column(Text)
    audio_url = Column(Text)
    transcript = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    claims = relationship(
        "Claim", back_populates="episode", cascade="all, delete-orphan"
    )
    quotes = relationship(
        "Quote", back_populates="episode", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<PodcastEpisode(id={self.id}, name='{self.name[:50]}...')>"


class Claim(Base):
    """
    Extracted factual claim with embedding and confidence score.

    Table: crypto.claims
    """

    __tablename__ = "claims"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(
        BigInteger,
        ForeignKey("crypto.podcast_episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    claim_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    embedding = Column(Vector(768))  # pgvector for similarity search
    confidence_components = Column(JSONB)
    reranker_scores = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    episode = relationship("PodcastEpisode", back_populates="claims")
    claim_quotes = relationship(
        "ClaimQuote", back_populates="claim", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Claim(id={self.id}, text='{self.claim_text[:50]}...', confidence={self.confidence:.2f})>"


class Quote(Base):
    """
    Text excerpt from transcript with position tracking.

    Table: crypto.quotes
    """

    __tablename__ = "quotes"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    episode_id = Column(
        BigInteger,
        ForeignKey("crypto.podcast_episodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    quote_text = Column(Text, nullable=False)
    start_position = Column(Integer)
    end_position = Column(Integer)
    speaker = Column(String(255))
    timestamp_seconds = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    episode = relationship("PodcastEpisode", back_populates="quotes")
    claim_quotes = relationship(
        "ClaimQuote", back_populates="quote", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Quote(id={self.id}, text='{self.quote_text[:50]}...', speaker='{self.speaker}')>"


class ClaimQuote(Base):
    """
    Many-to-many relationship between claims and quotes.

    Stores relevance scores and entailment validation results.

    Table: crypto.claim_quotes
    """

    __tablename__ = "claim_quotes"
    __table_args__ = {"schema": "crypto"}

    claim_id = Column(
        BigInteger,
        ForeignKey("crypto.claims.id", ondelete="CASCADE"),
        primary_key=True,
    )
    quote_id = Column(
        BigInteger,
        ForeignKey("crypto.quotes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    relevance_score = Column(Float, nullable=False)
    match_confidence = Column(Float, nullable=False)
    match_type = Column(String(20))
    entailment_score = Column(Float)
    entailment_relationship = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    claim = relationship("Claim", back_populates="claim_quotes")
    quote = relationship("Quote", back_populates="claim_quotes")

    def __repr__(self) -> str:
        return (
            f"<ClaimQuote(claim_id={self.claim_id}, quote_id={self.quote_id}, "
            f"relevance={self.relevance_score:.2f}, entailment='{self.entailment_relationship}')>"
        )
