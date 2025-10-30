# CLI Usage Guide

## Overview

The `process_episodes.py` CLI is the main entry point for processing podcast episodes through the claim extraction pipeline. It provides intelligent filtering, progress tracking, and comprehensive error handling.

## Quick Start

```bash
# Process all unprocessed episodes from all podcasts
uv run python -m src.cli.process_episodes

# Process specific podcast, limit to 5 episodes
uv run python -m src.cli.process_episodes --podcast-id 123 --limit 5

# Continue processing on errors
uv run python -m src.cli.process_episodes --continue-on-error

# Dry run (show what would be processed without processing)
uv run python -m src.cli.process_episodes --dry-run
```

## Command Line Arguments

### `--podcast-id <ID>`
**Optional** - Process episodes from a specific podcast

```bash
# Process only podcast 123
uv run python -m src.cli.process_episodes --podcast-id 123
```

**Default:** Process all podcasts

### `--limit <NUMBER>`
**Optional** - Maximum number of episodes to process

```bash
# Process only 10 episodes
uv run python -m src.cli.process_episodes --limit 10

# Process specific podcast, limit to 5
uv run python -m src.cli.process_episodes --podcast-id 123 --limit 5
```

**Default:** 0 (no limit, process all matching episodes)

### `--force`
**Optional** - Reprocess episodes that already have claims

```bash
# Reprocess all episodes (ignore existing claims)
uv run python -m src.cli.process_episodes --force

# Reprocess specific podcast
uv run python -m src.cli.process_episodes --podcast-id 123 --force
```

**Default:** Skip episodes that already have claims

**Use case:** Testing pipeline changes, updating claims after model improvements

### `--continue-on-error`
**Optional** - Continue processing if an episode fails

```bash
# Don't stop on errors
uv run python -m src.cli.process_episodes --continue-on-error
```

**Default:** Stop processing on first error

**Use case:** Batch processing where you want to process as many episodes as possible

### `--dry-run`
**Optional** - Show what would be processed without actually processing

```bash
# Preview what would be processed
uv run python -m src.cli.process_episodes --podcast-id 123 --limit 10 --dry-run
```

**Use case:** Testing filters, estimating processing time

## Processing Behavior

### Episode Selection

Episodes are processed in the following order:

1. **Filter by transcript:** Only episodes with transcripts
2. **Filter by podcast:** If `--podcast-id` specified
3. **Filter by processing status:** Skip processed episodes (unless `--force`)
4. **Order by date:** Newest first (`published_at DESC`, NULL dates last)
5. **Apply limit:** If `--limit` specified

### What is "Processed"?

An episode is considered **processed** if it has any claims in the database. The CLI skips these by default to avoid redundant work.

Use `--force` to reprocess episodes.

### Error Handling

**Default behavior:** Stop on first error
- Useful for debugging
- Ensures you see and fix issues immediately
- Exit code 1 on failure

**With `--continue-on-error`:** Continue processing
- Logs error and continues to next episode
- All errors shown in final summary
- Exit code 1 if any episodes failed
- Useful for batch processing

## Output Examples

### Processing Summary

```
╭─────────────────── Processing Summary ───────────────────╮
│ Setting              │ Value                             │
├──────────────────────┼───────────────────────────────────┤
│ Podcast ID           │ 123                               │
│ Episodes to process  │ 10                                │
│ Mode                 │ Skip processed                    │
│ Order                │ Newest first                      │
╰──────────────────────┴───────────────────────────────────╯

Episodes to process (first 5):
  1. Episode 789: Bitcoin Reaches New ATH (2025-10-20)
  2. Episode 788: Ethereum's Future (2025-10-18)
  3. Episode 787: DeFi Deep Dive (2025-10-15)
  4. Episode 786: NFT Markets (2025-10-12)
  5. Episode 785: Crypto Regulation (2025-10-10)
  ... and 5 more
```

### Progress Bar

```
⠋ Processing episode 3/10: Bitcoin Reaches New ATH... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 30% 0:01:23 0:03:12
```

### Episode Results

```
Episode 3/10: Bitcoin Reaches New ATH (ID: 789)
Metric              Value
──────────────────  ────────────────────────────────
Claims              20 → 15 → 12 saved
Quotes              45 → 32 (entailment) → 28 saved
Entailment filtered 13 quotes
Duplicates merged   3 claims
Time                48.3s
```

### Final Statistics

```
═══════════════════════════════════════════════════════════════════
PROCESSING COMPLETE
═══════════════════════════════════════════════════════════════════

╭──────────────────── Final Statistics ────────────────────╮
│ Metric                │ Value                           │
├───────────────────────┼─────────────────────────────────┤
│ Episodes processed    │ 18/20 (90%)                     │
│ Failed                │ 2                               │
│ Claims saved          │ 234                             │
│ Quotes saved          │ 567                             │
│ Duplicates merged     │ 12                              │
│ Total time            │ 15.4 minutes                    │
│ Average per episode   │ 51.3s                           │
╰───────────────────────┴─────────────────────────────────╯

Failed Episodes:
  • Episode 456: Failed Episode Name
    Error: Database connection timeout
  • Episode 789: Another Failed Episode
    Error: Transcript parsing error
```

## Common Use Cases

### Initial Backfill

Process all historical episodes:

```bash
# Process all podcasts (recommended: start with small limit to test)
uv run python -m src.cli.process_episodes --limit 10

# If successful, process all
uv run python -m src.cli.process_episodes --continue-on-error
```

### Regular Processing

Process new episodes daily:

```bash
# Process latest 10 episodes (skips already-processed)
uv run python -m src.cli.process_episodes --limit 10
```

### Reprocessing After Model Updates

Reprocess episodes with updated models:

```bash
# Reprocess specific podcast with new entailment model
uv run python -m src.cli.process_episodes --podcast-id 123 --force --continue-on-error
```

### Testing

Test processing on a few episodes:

```bash
# Dry run to see what would be processed
uv run python -m src.cli.process_episodes --podcast-id 123 --limit 5 --dry-run

# Actually process
uv run python -m src.cli.process_episodes --podcast-id 123 --limit 5
```

### Error Recovery

Continue processing after fixing errors:

```bash
# Process with error handling
uv run python -m src.cli.process_episodes --continue-on-error

# Review errors in logs
tail -f logs/extraction_*.log
```

## Exit Codes

- **0** - Success (all episodes processed)
- **1** - Failure (one or more episodes failed)

**Note:** With `--continue-on-error`, exit code 1 is returned if ANY episode failed, even if others succeeded.

## Logging

All processing is logged to:
```
logs/extraction_YYYYMMDD_HHMMSS.log
```

**Log levels:**
- INFO: Processing progress
- DEBUG: Detailed operation info
- ERROR: Failures and exceptions

## Tips

1. **Start small:** Test with `--limit 5` before processing all episodes
2. **Use dry-run:** Preview what will be processed with `--dry-run`
3. **Monitor logs:** Watch logs in real-time with `tail -f logs/extraction_*.log`
4. **Continue on errors:** Use `--continue-on-error` for batch processing
5. **Check database:** Verify claims were saved with database queries

## Troubleshooting

### No episodes to process

**Cause:** All episodes already processed or no episodes with transcripts

**Solution:**
- Use `--force` to reprocess
- Check database for existing claims
- Verify episodes have transcripts

### Processing hangs

**Cause:** LLM service (Ollama) not responding

**Solution:**
- Check Ollama is running: `ollama list`
- Restart Ollama: `ollama serve`
- Check network connectivity

### Database errors

**Cause:** PostgreSQL not available or pgvector not installed

**Solution:**
- Check PostgreSQL is running
- Verify DATABASE_URL in .env
- Ensure pgvector extension is installed

### Reranker errors

**Cause:** Reranker Docker container not running

**Solution:**
```bash
docker-compose -f docker-compose.reranker.yml up -d
curl http://localhost:8080/health
```

## Performance

**Typical processing time:**
- ~45-60 seconds per episode
- ~60-80 episodes per hour (single-threaded)

**Bottlenecks:**
- LLM calls (claim extraction + entailment)
- Reranker calls (quote ranking + deduplication)
- Database queries (similarity search)

**Optimization tips:**
- Process during off-peak hours
- Use `--limit` for incremental processing
- Monitor resource usage (CPU, memory, disk I/O)
