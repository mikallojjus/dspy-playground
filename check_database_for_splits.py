"""
Check database for claims that may have been affected by the split bug.

This script queries the database to find claims that start with "s " which
indicates they were likely split by the json_repair bug.
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, '.')

import asyncio
from src.database.connection import get_db_session
from src.database.models import Claim
from sqlalchemy import func


def check_database_for_split_claims():
    """Query database for claims that start with 's ' pattern."""

    print("=" * 80)
    print("Checking Database for Split Claims")
    print("=" * 80)
    print()

    session = get_db_session()

    try:
        # First, get total count
        print("Counting total claims in database...")
        total_claims = session.query(func.count(Claim.id)).scalar()
        print(f"Total claims in database: {total_claims:,}")
        print()

        # Find claims starting with "s " (lowercase s followed by space)
        print("Searching for claims starting with 's '...")
        split_claims = session.query(Claim).filter(
            Claim.claim_text.like('s %')
        ).limit(100).all()  # Limit to first 100 for speed

        print(f"Found {len(split_claims)} claims starting with 's ':\n")

        if split_claims:
            for i, claim in enumerate(split_claims, 1):
                print(f"{i}. Episode: {claim.episode_id}")
                print(f"   Claim ID: {claim.id}")
                print(f"   Text: {claim.claim_text[:100]}...")
                print(f"   Confidence: {claim.confidence}")
                print()
        else:
            print("✓ No split claims found! Database is clean.")
            print()

        # Also check for very short claims (< 15 chars) as potential first parts
        print("-" * 80)
        print()

        print("Searching for very short claims...")
        short_claims = session.query(Claim).filter(
            func.length(Claim.claim_text) < 15
        ).limit(100).all()  # Limit to first 100 for speed

        print(f"Found {len(short_claims)} very short claims (< 15 characters):\n")

        if short_claims:
            for i, claim in enumerate(short_claims, 1):
                print(f"{i}. Episode: {claim.episode_id}")
                print(f"   Claim ID: {claim.id}")
                print(f"   Text: '{claim.claim_text}'")
                print(f"   Length: {len(claim.claim_text)} chars")
                print()
        else:
            print("✓ No suspiciously short claims found.")
            print()

        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        print(f"Total claims starting with 's ': {len(split_claims)}")
        print(f"Total very short claims: {len(short_claims)}")
        print()

        if split_claims or short_claims:
            print("⚠ Some potentially affected claims were found.")
            print()
            print("These claims were likely created before the fix was applied.")
            print("They will not occur in new extractions with the fix in place.")
            print()
            print("To clean existing data, you may want to:")
            print("1. Re-run extraction on affected episodes")
            print("2. Or manually review and fix these specific claims")
        else:
            print("✓ Database looks clean! No affected claims found.")

        print()

    finally:
        session.close()


if __name__ == "__main__":
    check_database_for_split_claims()
