# Implementation Plan: Update People Extraction Script to Mark Source

## Overview
Update the guest extraction system to mark in-house extracted people with `data_source = "guest-extraction-api"` in the PostgreSQL production database.

## Database Schema (Discovered)

### `crypto.people`
| Column | Type | Notes |
|--------|------|-------|
| id | bigint | Primary key, auto-increment |
| name | text | NOT NULL |
| x_url | text | Twitter/X URL |
| linkedin_url | text | LinkedIn URL |
| instagram_url | text | Instagram URL |
| facebook_url | text | Facebook URL |
| github_url | text | GitHub URL |
| youtube_url | text | YouTube URL |
| telegram_url | text | Telegram URL |
| tiktok_url | text | TikTok URL |
| medium_url | text | Medium URL |
| website | text | Personal website |
| wikipedia_url | text | Wikipedia URL |
| avatar | text | Avatar image URL |
| description | text | Person description |
| **data_source** | text | **Source tracking field** |
| created_at | timestamptz | Default: CURRENT_TIMESTAMP |
| updated_at | timestamptz | Default: CURRENT_TIMESTAMP |

### `crypto.podcast_credits`
| Column | Type | Notes |
|--------|------|-------|
| id | bigint | Primary key |
| person_id | bigint | FK to people.id |
| episode_id | bigint | FK to podcast_episodes.id |
| role | varchar | Default: 'guest' |
| created_at | timestamptz | Default: CURRENT_TIMESTAMP |

### Current Data Source Distribution
- `NULL`: 17,265 people
- `podchaser`: 57 people

---

## Implementation Steps

### Step 1: Add People Model
**File**: `src/database/models.py`

```python
class Person(Base):
    """Person extracted from podcast episodes."""
    __tablename__ = "people"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    x_url = Column(Text)
    linkedin_url = Column(Text)
    instagram_url = Column(Text)
    facebook_url = Column(Text)
    github_url = Column(Text)
    youtube_url = Column(Text)
    telegram_url = Column(Text)
    tiktok_url = Column(Text)
    medium_url = Column(Text)
    website = Column(Text)
    wikipedia_url = Column(Text)
    avatar = Column(Text)
    description = Column(Text)
    data_source = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PodcastCredit(Base):
    """Links people to podcast episodes with roles."""
    __tablename__ = "podcast_credits"
    __table_args__ = {"schema": "crypto"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    person_id = Column(BigInteger, ForeignKey("crypto.people.id"), nullable=False)
    episode_id = Column(BigInteger, ForeignKey("crypto.podcast_episodes.id"), nullable=False)
    role = Column(String(50), nullable=False, default="guest")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
```

### Step 2: Create Person Repository
**File**: `src/database/person_repository.py` (new)

```python
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from src.database.models import Person, PodcastCredit
from src.database.connection import get_session

DATA_SOURCE_GUEST_EXTRACTION = "guest-extraction-api"

# URL field mapping from extracted URLs to database columns
URL_FIELD_MAP = {
    "twitter.com": "x_url",
    "x.com": "x_url",
    "linkedin.com": "linkedin_url",
    "instagram.com": "instagram_url",
    "facebook.com": "facebook_url",
    "github.com": "github_url",
    "youtube.com": "youtube_url",
    "t.me": "telegram_url",
    "telegram.me": "telegram_url",
    "tiktok.com": "tiktok_url",
    "medium.com": "medium_url",
}

def save_extracted_guests(
    guests: List[Dict[str, Any]],
    episode_id: int,
    source: str = DATA_SOURCE_GUEST_EXTRACTION
) -> List[Person]:
    """
    Save extracted guests to database with source tracking.
    Creates Person records and PodcastCredit entries.
    """
    # Implementation details...
```

### Step 3: Update Guest Extraction Service
**File**: `src/api/services/guest_extraction_service.py`

Add optional persistence with source tracking:
- Add `episode_id` and `persist` parameters
- When `persist=True`, save to database with `data_source = "guest-extraction-api"`

### Step 4: Update API Endpoint & Schema
**Files**:
- `src/api/routers/guest_extraction.py`
- `src/api/schemas/guest_extraction_schema.py`

Add:
- `episode_id` field to request schema
- `persist` query parameter to endpoint

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `src/database/models.py` | Edit | Add Person and PodcastCredit models |
| `src/database/person_repository.py` | Create | New repository for person DB operations |
| `src/api/services/guest_extraction_service.py` | Edit | Add persistence with source tracking |
| `src/api/routers/guest_extraction.py` | Edit | Add persist parameter |
| `src/api/schemas/guest_extraction_schema.py` | Edit | Add episode_id field |

---

## Key Implementation Details

1. **Source Value**: `data_source = "guest-extraction-api"`
2. **URL Parsing**: Map extracted URLs to appropriate columns (x_url, linkedin_url, etc.)
3. **Deduplication**: Check if person already exists by name before creating
4. **Credit Creation**: Create PodcastCredit entry linking person to episode with role="guest"
5. **Backward Compatible**: Persistence is optional (default: False)

---

## Testing Strategy

1. Unit test: `save_extracted_guests()` with mocked DB
2. Integration test: API with `persist=True`
3. Verify: `SELECT * FROM crypto.people WHERE data_source = 'guest-extraction-api'`
