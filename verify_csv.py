#!/usr/bin/env python3
"""
Script to verify if the Notion-exported CSV can be properly parsed.
Checks for column consistency, proper parsing, and data integrity.
"""

import csv
import sys
from pathlib import Path


def verify_csv(file_path: str):
    """Verify CSV file can be properly parsed."""

    print(f"Verifying CSV file: {file_path}\n")
    print("=" * 80)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to read with csv.reader which handles quoted fields properly
            reader = csv.DictReader(f)

            # Get expected columns from header
            expected_columns = reader.fieldnames
            print(f"✓ Header found with {len(expected_columns)} columns:")
            for i, col in enumerate(expected_columns, 1):
                print(f"  {i}. {col}")
            print()

            # Parse all rows and check consistency
            rows = []
            column_mismatches = []
            parsing_errors = []

            for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
                try:
                    # Check if number of columns matches
                    actual_columns = len(row)

                    if actual_columns != len(expected_columns):
                        column_mismatches.append({
                            'row': row_num,
                            'expected': len(expected_columns),
                            'actual': actual_columns
                        })

                    rows.append(row)

                except Exception as e:
                    parsing_errors.append({
                        'row': row_num,
                        'error': str(e)
                    })

            print(f"✓ Successfully parsed {len(rows)} rows")
            print()

            # Report statistics
            print("=" * 80)
            print("STATISTICS")
            print("=" * 80)

            # Check for empty cells in each column
            for col in expected_columns:
                non_empty = sum(1 for row in rows if row.get(col, '').strip())
                empty = len(rows) - non_empty
                print(f"{col}:")
                print(f"  Non-empty: {non_empty}")
                print(f"  Empty: {empty}")

            print()

            # Show sample data from first few rows
            print("=" * 80)
            print("SAMPLE DATA (first 3 rows)")
            print("=" * 80)

            for i, row in enumerate(rows[:3], 1):
                print(f"\nRow {i}:")
                for col in expected_columns:
                    value = row.get(col, '')
                    # Truncate long values for display
                    display_value = value[:100] + "..." if len(value) > 100 else value
                    display_value = display_value.replace('\n', '\\n')
                    print(f"  {col}: {display_value}")

            print()

            # Report issues
            print("=" * 80)
            print("ISSUES")
            print("=" * 80)

            if column_mismatches:
                print(f"⚠ Found {len(column_mismatches)} rows with column count mismatches:")
                for mismatch in column_mismatches[:10]:  # Show first 10
                    print(f"  Row {mismatch['row']}: expected {mismatch['expected']} columns, got {mismatch['actual']}")
                if len(column_mismatches) > 10:
                    print(f"  ... and {len(column_mismatches) - 10} more")
            else:
                print("✓ No column count mismatches found")

            print()

            if parsing_errors:
                print(f"✗ Found {len(parsing_errors)} parsing errors:")
                for error in parsing_errors[:10]:  # Show first 10
                    print(f"  Row {error['row']}: {error['error']}")
                if len(parsing_errors) > 10:
                    print(f"  ... and {len(parsing_errors) - 10} more")
            else:
                print("✓ No parsing errors found")

            print()

            # Final verdict
            print("=" * 80)
            print("VERDICT")
            print("=" * 80)

            if not column_mismatches and not parsing_errors:
                print("✓ CSV file is properly formatted and can be parsed successfully!")
                return 0
            else:
                print("⚠ CSV file has some issues (see above)")
                return 1

    except FileNotFoundError:
        print(f"✗ Error: File not found: {file_path}")
        return 1
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    csv_file = "data/claims_from_chunks_reviewed.csv"

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    exit_code = verify_csv(csv_file)
    sys.exit(exit_code)
