"""
CLI tools for the claim extraction pipeline.

This module provides command-line interfaces for processing podcast episodes
and managing the extraction pipeline.

Modules:
    episode_query: Query logic for finding episodes to process
    process_episodes: Main CLI for processing episodes

Usage:
    Run the CLI directly as a module:

    uv run python -m src.cli.process_episodes --help
"""

# Don't import modules here to avoid conflicts when running as -m
# CLI scripts should be run directly, not imported
__all__ = []
