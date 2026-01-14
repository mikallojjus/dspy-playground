# Implementation Plan: Update People Extraction Script to Mark Source

## Status: ✅ COMPLETED

The fix has already been implemented in the **pg-migrations** repository.

---

## Summary

**Task**: Mark in-house extracted people's source in PostgreSQL production database.

**Solution**: Set `data_source = "extraction"` when saving guests extracted via the guest-extraction-api.

---

## Architecture

```
┌─────────────────────────┐      JSON       ┌─────────────────────────┐
│   dspy-playground       │ ──────────────► │   pg-migrations         │
│   (this repo)           │                 │                         │
│                         │                 │                         │
│   POST /guests          │                 │   importGuestsForEpisodes()
│   - extracts guests     │                 │   - saves to DB         │
│   - returns JSON        │                 │   - sets data_source    │
└─────────────────────────┘                 └─────────────────────────┘
```

---

## Implementation (Already Done)

**Repository**: `pg-migrations`
**File**: `src/etl/podcasts/operations/extraction-import-operations.ts`
**Method**: `importGuestsForEpisodes()`
**Change**: `newPerson.data_source = "extraction";`

**Deployed**: Recently added (as of yesterday)

---

## Database Schema Reference

### `crypto.people`
| Column | Type | Notes |
|--------|------|-------|
| id | bigint | Primary key |
| name | text | NOT NULL |
| data_source | text | Source tracking: `"extraction"`, `"podchaser"`, etc. |
| ... | ... | Other URL and metadata fields |

### `crypto.podcast_credits`
| Column | Type | Notes |
|--------|------|-------|
| person_id | bigint | FK to people.id |
| episode_id | bigint | FK to podcast_episodes.id |
| role | varchar | Default: 'guest' |

### Current Data Source Distribution (before deployment)
- `NULL`: 17,265 people (legacy, pre-tracking)
- `podchaser`: 57 people (external source)
- `extraction`: 0 people (newly deployed, will populate going forward)

---

## No Changes Required in This Repository

The `dspy-playground` guest extraction API (`POST /guests`) correctly returns JSON with extracted guest data. Database persistence and source tracking are handled by `pg-migrations`.
