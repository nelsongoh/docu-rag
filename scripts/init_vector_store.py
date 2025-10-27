#!/usr/bin/env python3
"""
Initialize Vector Store - Vectorize processed chunks and populate collections

This script orchestrates the vectorization process for all configured collections.
It loads processed chunks, creates vector stores, and populates them with embeddings.

Usage:
    python scripts/init_vector_store.py [--config CONFIG_PATH] [--data DATA_DIR] [--db DB_PATH]

Options:
    --config CONFIG_PATH    Path to vector_store_config.json (default: config/vector_store_config.json)
    --data DATA_DIR         Directory with processed chunks (default: data/processed)
    --db DB_PATH            ChromaDB storage path (default: ./chroma_db)

Examples:
    python scripts/init_vector_store.py
    python scripts/init_vector_store.py --config config/vector_store_config.json --data data/processed
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import vectorize_collections, VectorStoreError


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Initialize vector store by vectorizing processed chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to vector_store_config.json (default: config/vector_store_config.json)"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed",
        help="Directory containing processed chunks (default: data/processed)"
    )

    parser.add_argument(
        "--db",
        type=str,
        default="./chroma_db",
        help="ChromaDB storage path (default: ./chroma_db)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    try:
        print("=" * 60)
        print("VECTOR STORE INITIALIZATION")
        print("=" * 60)

        # Call vectorize_collections function
        results = vectorize_collections(
            config_path=args.config,
            data_dir=args.data,
            chroma_db_path=args.db
        )

        # Print results
        print("\n" + "=" * 60)
        print("INITIALIZATION COMPLETE")
        print("=" * 60)
        print(f"Total collections: {results['total']}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")

        # Exit with appropriate code
        if results['failed']:
            print("\nSome collections failed. Please review the errors above.")
            return 1
        elif results['skipped'] and not results['successful']:
            print("\nNo collections were successfully created.")
            return 1
        else:
            print("\nVector store initialization successful!")
            return 0

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user (Ctrl+C)")
        return 130

    except VectorStoreError as e:
        print(f"\nERROR: {e}")
        return 1

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())