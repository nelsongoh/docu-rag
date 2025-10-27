import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAIError
import json
import tiktoken
import sys
from typing import List, Dict, Optional
from pathlib import Path
import os
import time
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config.constants import (
    MAX_TOKENS_PER_CHUNK,
    VALID_EMBEDDING_MODELS,
    EMBEDDING_PRICING,
    PRICING_DATE,
    TOKENIZER_ENCODING
)

class VectorStoreError(Exception):
    """Custom exception for VectorStore errors"""
    pass


class VectorStoreSingleton:
    """
    Singleton pattern implementation for VectorStore to avoid costly re-initialization.

    Ensures only one VectorStore instance is created per collection and reused
    throughout the application lifecycle. This prevents redundant initialization
    which can be costly in terms of API calls and disk I/O.

    Usage:
        vector_store = VectorStoreSingleton.get_instance(
            collection_name="my_collection",
            description="My collection description"
        )
    """
    _instances: Dict[str, 'VectorStore'] = {}
    _lock = object()

    @classmethod
    def get_instance(
        cls,
        collection_name: str,
        description: str = "",
        metadata: Optional[Dict] = None,
        chroma_db_path: str = "./chroma_db"
    ) -> 'VectorStore':
        """
        Get or create a VectorStore instance for the given collection.

        Args:
            collection_name: Name of the ChromaDB collection
            description: Description of the collection
            metadata: Metadata to attach to the collection
            chroma_db_path: Path to ChromaDB storage directory

        Returns:
            VectorStore instance (singleton per collection)

        Raises:
            ValueError: If collection_name is invalid
        """
        if not collection_name or not isinstance(collection_name, str):
            raise ValueError("collection_name must be a non-empty string")

        if collection_name not in cls._instances:
            if metadata is None:
                metadata = {}

            cls._instances[collection_name] = VectorStore(
                collection_name=collection_name,
                description=description,
                metadata=metadata,
                chroma_db_path=chroma_db_path
            )

        return cls._instances[collection_name]

    @classmethod
    def reset(cls) -> None:
        """
        Clear all cached instances.

        Use only for testing or cleanup purposes. This will force re-initialization
        of all instances on the next get_instance() call.
        """
        cls._instances.clear()

class VectorStore:
    def __init__(self, collection_name: str, description: str, metadata: Dict, chroma_db_path: str = "./chroma_db"):
        """Initialize vector store with configurable collection

        Args:
            collection_name: Name of the ChromaDB collection
            description: Description of the collection
            metadata: Metadata to attach to the collection
            chroma_db_path: Path to ChromaDB storage directory

        Raises:
            VectorStoreError: If API key is missing or ChromaDB initialization fails
        """
        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise VectorStoreError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        # Initialize ChromaDB (persists to disk)
        try:
            self.client = chromadb.PersistentClient(path=chroma_db_path)
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}")

        self.collection_name = collection_name
        self.openai_client = openai.OpenAI(api_key=api_key)

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize tokenizer: {e}")

        # Create or get collection with dynamic metadata
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "description": description,
                    **metadata
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection '{collection_name}': {e}")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def embed_text(self, text: str, model: str = "text-embedding-3-small", retry_count: int = 3) -> List[float]:
        """Generate embedding using OpenAI with automatic token limit handling

        Args:
            text: Text to embed
            model: OpenAI embedding model to use
            retry_count: Number of retries for rate limit errors

        Returns:
            List of embedding values

        Raises:
            VectorStoreError: If model is invalid or API call fails
        """
        # Validate model
        if model not in VALID_EMBEDDING_MODELS:
            raise VectorStoreError(
                f"Invalid embedding model: '{model}'. "
                f"Valid models: {', '.join(sorted(VALID_EMBEDDING_MODELS))}"
            )

        # Validate text
        if not text or not text.strip():
            raise VectorStoreError("Cannot embed empty text")

        # Check token count and truncate if necessary
        token_count = self.count_tokens(text)
        if token_count > MAX_TOKENS_PER_CHUNK:
            print(f"WARNING: Text has {token_count:,} tokens, truncating to {MAX_TOKENS_PER_CHUNK:,}")
            text = self.truncate_text(text, MAX_TOKENS_PER_CHUNK)

        # Retry logic for rate limiting
        for attempt in range(retry_count):
            try:
                response = self.openai_client.embeddings.create(
                    model=model,
                    input=text
                )
                return response.data[0].embedding

            except OpenAIError as e:
                error_str = str(e)

                # Handle rate limiting
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < retry_count - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                        print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 2}/{retry_count}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise VectorStoreError(f"Rate limit exceeded after {retry_count} retries: {e}")

                # Handle token limit errors (shouldn't happen due to pre-check, but just in case)
                elif "max_tokens" in error_str.lower():
                    raise VectorStoreError(
                        f"Token limit error despite pre-check. Text may be too large. "
                        f"Try reducing chunk size in data_processor.py. Error: {e}"
                    )

                # Other OpenAI errors
                else:
                    raise VectorStoreError(f"OpenAI API error: {e}")

            except Exception as e:
                raise VectorStoreError(f"Unexpected error during embedding: {e}")

        raise VectorStoreError(f"Failed to generate embedding after {retry_count} attempts")

    def get_existing_chunk_ids(self) -> set:
        """Get IDs of chunks already in the collection

        Returns:
            Set of existing chunk IDs
        """
        try:
            # Get all items from collection
            result = self.collection.get()
            existing_ids = set(result['ids'])
            return existing_ids
        except Exception as e:
            print(f"WARNING: Could not retrieve existing chunks: {e}")
            return set()

    def add_documents(self, chunks: List[Dict], embedding_model: str = "text-embedding-3-small", resume: bool = True):
        """Add document chunks to vector store with resume capability

        Args:
            chunks: List of document chunks to add
            embedding_model: OpenAI embedding model to use
            resume: If True, skip chunks that already exist in the collection

        Raises:
            VectorStoreError: If chunks are invalid or operations fail
        """
        # Validate inputs
        if not chunks:
            raise VectorStoreError("Cannot add empty chunks list")

        if embedding_model not in VALID_EMBEDDING_MODELS:
            raise VectorStoreError(
                f"Invalid embedding model: '{embedding_model}'. "
                f"Valid models: {', '.join(sorted(VALID_EMBEDDING_MODELS))}"
            )

        # Check for existing chunks if resume is enabled
        existing_ids = set()
        if resume:
            print("Checking for existing chunks...")
            existing_ids = self.get_existing_chunk_ids()
            if existing_ids:
                print(f"Found {len(existing_ids)} existing chunks in collection")
            else:
                print("No existing chunks found, starting fresh")

        print(f"Processing {len(chunks)} total chunks...")

        # Track statistics
        skipped_count = 0
        processed_count = 0
        error_count = 0

        for i, chunk in enumerate(chunks):
            try:
                # Validate chunk structure
                if 'content' not in chunk:
                    print(f"WARNING: Skipping chunk {i} - missing 'content' field")
                    error_count += 1
                    continue

                if 'chunk_id' not in chunk:
                    print(f"WARNING: Skipping chunk {i} - missing 'chunk_id' field")
                    error_count += 1
                    continue

                chunk_id = f"chunk_{chunk['chunk_id']}"

                # Skip if already exists
                if resume and chunk_id in existing_ids:
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"Skipped {skipped_count} existing chunks...")
                    continue

                # Generate embedding
                embedding = self.embed_text(chunk['content'], model=embedding_model)

                # Add to collection
                try:
                    self.collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk['content']],
                        metadatas=[{
                            'title': chunk.get('title', ''),
                            'chunk_id': str(chunk['chunk_id']),
                            **chunk.get('metadata', {})
                        }]
                    )
                    processed_count += 1

                except Exception as add_error:
                    # Handle duplicate ID errors
                    error_str = str(add_error).lower()
                    if 'already exists' in error_str or 'duplicate' in error_str or 'unique constraint' in error_str:
                        # This chunk already exists, count as skipped
                        skipped_count += 1
                        if skipped_count % 100 == 0:
                            print(f"Skipped {skipped_count} existing chunks (including duplicates)...")
                        continue
                    else:
                        # Some other error
                        raise

                if processed_count % 10 == 0:
                    total_done = skipped_count + processed_count + error_count
                    print(f"Progress: {total_done}/{len(chunks)} chunks "
                          f"(New: {processed_count}, Skipped: {skipped_count}, Errors: {error_count})")

            except VectorStoreError:
                # Re-raise VectorStore errors (API issues, validation, etc.)
                raise
            except Exception as e:
                error_str = str(e).lower()
                # Handle duplicate errors that weren't caught above
                if 'already exists' in error_str or 'duplicate' in error_str or 'unique constraint' in error_str:
                    skipped_count += 1
                    if skipped_count % 100 == 0:
                        print(f"Skipped {skipped_count} existing chunks (including duplicates)...")
                    continue
                else:
                    print(f"ERROR: Failed to process chunk {i} (ID: {chunk.get('chunk_id', 'unknown')}): {e}")
                    print(f"Continuing with next chunk...")
                    error_count += 1
                    continue

        # Final summary
        print(f"\n{'=' * 60}")
        print(f"✓ Vector store population complete")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Newly added: {processed_count}")
        print(f"  Skipped (already exist): {skipped_count}")
        print(f"  Errors: {error_count}")
        print(f"{'=' * 60}")

    def search(self, query: str, n_results: int = 5, embedding_model: str = "text-embedding-3-small") -> Dict:
        """Search for relevant chunks

        Args:
            query: Search query
            n_results: Number of results to return
            embedding_model: OpenAI embedding model to use

        Returns:
            Dictionary with documents, metadatas, and distances
        """
        query_embedding = self.embed_text(query, model=embedding_model)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }


def estimate_cost(num_chunks: int, model: str = "text-embedding-3-small") -> tuple[float, str]:
    """Estimate cost of embedding chunks

    Args:
        num_chunks: Number of chunks to embed
        model: OpenAI embedding model

    Returns:
        Tuple of (estimated cost in USD, pricing date)
    """
    # Pricing per 1M tokens (as of January 2025)
    pricing_date = "2025-01-24"
    pricing = {
        "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
        "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
    }

    # Rough estimate: 500 tokens per chunk average
    estimated_tokens = num_chunks * 500
    cost_per_million = pricing.get(model, 0.02)
    estimated_cost = (estimated_tokens / 1_000_000) * cost_per_million

    return estimated_cost, pricing_date


def confirm_execution(num_chunks: int, collections: List[str], model: str) -> bool:
    """Ask user for confirmation before proceeding with API calls

    Args:
        num_chunks: Total number of chunks to process
        collections: List of collection names to create
        model: Embedding model to use

    Returns:
        True if user confirms, False otherwise
    """
    estimated_cost, pricing_date = estimate_cost(num_chunks, model)

    print("\n" + "=" * 60)
    print("VECTOR STORE CREATION CONFIRMATION")
    print("=" * 60)
    print(f"Collections to create: {len(collections)}")
    for collection in collections:
        print(f"  - {collection}")
    print(f"\nTotal documents to vectorize: {num_chunks:,}")
    print(f"Embedding model: {model}")
    print(f"Estimated cost: ${estimated_cost:.4f} USD")
    print(f"Pricing as of: {pricing_date}")
    print("=" * 60)

    response = input("\nProceed with vectorization? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def validate_config(config: Dict) -> None:
    """Validate configuration structure

    Args:
        config: Configuration dictionary

    Raises:
        VectorStoreError: If configuration is invalid
    """
    if 'collections' not in config:
        raise VectorStoreError("Configuration missing 'collections' key")

    if not config['collections']:
        raise VectorStoreError("Configuration 'collections' list is empty")

    if not isinstance(config['collections'], list):
        raise VectorStoreError("Configuration 'collections' must be a list")

    # Validate embedding model
    embedding_model = config.get('embedding_model', 'text-embedding-3-small')
    if embedding_model not in VALID_EMBEDDING_MODELS:
        raise VectorStoreError(
            f"Invalid embedding model in config: '{embedding_model}'. "
            f"Valid models: {', '.join(sorted(VALID_EMBEDDING_MODELS))}"
        )

    # Validate each collection config
    for i, coll in enumerate(config['collections']):
        if not isinstance(coll, dict):
            raise VectorStoreError(f"Collection {i} is not a dictionary")

        required_fields = ['name', 'source', 'description']
        for field in required_fields:
            if field not in coll:
                raise VectorStoreError(
                    f"Collection {i} missing required field: '{field}'"
                )

        if not coll['name'] or not isinstance(coll['name'], str):
            raise VectorStoreError(f"Collection {i} has invalid 'name' field")

        if not coll['source'] or not isinstance(coll['source'], str):
            raise VectorStoreError(f"Collection {i} has invalid 'source' field")


def load_processed_chunks(processed_dir: str = "data/processed") -> Dict[str, List[Dict]]:
    """Load all processed chunk files from directory

    Args:
        processed_dir: Directory containing processed chunk files

    Returns:
        Dictionary mapping filenames to their chunks
    """
    processed_path = Path(processed_dir)

    if not processed_path.exists():
        print(f"Processed directory not found: {processed_dir}")
        return {}

    # Find all chunked JSON files
    chunk_files = list(processed_path.glob('chunked_*.json'))

    if not chunk_files:
        print(f"No chunked files found in {processed_dir}")
        return {}

    loaded_chunks = {}

    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)
                loaded_chunks[chunk_file.name] = chunks
                print(f"Loaded {len(chunks)} chunks from {chunk_file.name}")
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {chunk_file.name}: {e}")
        except Exception as e:
            print(f"ERROR: Failed to load {chunk_file.name}: {e}")

    return loaded_chunks


def vectorize_collections(
    config_path: str = None,
    data_dir: str = "data/processed",
    chroma_db_path: str = "./chroma_db"
) -> Dict[str, any]:
    """
    Vectorize processed chunks and populate vector store collections.

    This function orchestrates the vectorization process for all configured
    collections. It loads processed chunks, creates vector stores, and adds
    documents with error handling and user confirmation.

    Args:
        config_path: Path to vector_store_config.json. If None, uses default location.
        data_dir: Directory containing processed chunk files
        chroma_db_path: Path to ChromaDB storage directory

    Returns:
        Dictionary with vectorization results:
        {
            'successful': List[str],  # Collection names successfully created
            'failed': List[str],      # Collection names that failed
            'skipped': List[str],     # Collection names that were skipped
            'total': int              # Total collections attempted
        }

    Raises:
        VectorStoreError: If configuration is invalid or critical operations fail
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'vector_store_config.json'
    else:
        config_path = Path(config_path)

    # Load and validate configuration
    if not config_path.exists():
        raise VectorStoreError(
            f"Configuration file not found at {config_path}. "
            "Please create config/vector_store_config.json"
        )

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise VectorStoreError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise VectorStoreError(f"Failed to load configuration: {e}")

    # Validate configuration
    validate_config(config)

    # Load all processed chunks
    print("Loading processed chunks...")
    all_chunks = load_processed_chunks(data_dir)

    if not all_chunks:
        raise VectorStoreError(
            f"No processed chunks found in {data_dir}. "
            "Please run data_processor.py first."
        )

    # Calculate total chunks
    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    collection_names = [coll['name'] for coll in config['collections']]
    embedding_model = config.get('embedding_model', 'text-embedding-3-small')

    # Ask for user confirmation
    if not confirm_execution(total_chunks, collection_names, embedding_model):
        print("\nOperation cancelled by user.")
        return {
            'successful': [],
            'failed': [],
            'skipped': collection_names,
            'total': len(collection_names)
        }

    print("\nProceeding with vectorization...")

    # Track success/failure
    total_collections = len(config['collections'])
    successful_collections = []
    failed_collections = []
    skipped_collections = []

    # Process each collection
    for collection_config in config['collections']:
        collection_name = collection_config['name']
        source = collection_config['source']
        description = collection_config['description']
        metadata = collection_config.get('metadata', {})

        print(f"\n{'=' * 60}")
        print(f"Creating collection: {collection_name}")
        print(f"{'=' * 60}")

        # Find matching chunked file based on source
        matching_file = None
        matching_chunks = None

        for filename, chunks in all_chunks.items():
            # Look for files that match the source name
            if source in filename.lower():
                matching_file = filename
                matching_chunks = chunks
                break

        if not matching_chunks:
            print(f"WARNING: No processed chunks found for source '{source}'")
            print(f"Skipping collection '{collection_name}'")
            skipped_collections.append(collection_name)
            continue

        print(f"Using chunks from: {matching_file}")
        print(f"Number of chunks: {len(matching_chunks)}")

        try:
            # Create vector store
            vector_store = VectorStore(
                collection_name=collection_name,
                description=description,
                metadata=metadata,
                chroma_db_path=chroma_db_path
            )

            # Add documents
            vector_store.add_documents(matching_chunks, embedding_model=embedding_model)
            successful_collections.append(collection_name)

        except VectorStoreError as e:
            print(f"ERROR: Failed to process collection '{collection_name}': {e}")
            print("Continuing with next collection...")
            failed_collections.append(collection_name)
            continue

    # Print summary
    print(f"\n{'=' * 60}")
    print("VECTORIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total collections: {total_collections}")
    print(f"Successful: {len(successful_collections)}")
    print(f"Failed: {len(failed_collections)}")
    print(f"Skipped: {len(skipped_collections)}")

    if successful_collections:
        print(f"\nSuccessfully created collections:")
        for name in successful_collections:
            print(f"  ✓ {name}")

    if failed_collections:
        print(f"\nFailed collections:")
        for name in failed_collections:
            print(f"  ✗ {name}")

    if skipped_collections:
        print(f"\nSkipped collections:")
        for name in skipped_collections:
            print(f"  - {name}")

    print(f"{'=' * 60}")

    return {
        'successful': successful_collections,
        'failed': failed_collections,
        'skipped': skipped_collections,
        'total': total_collections
    }


# DEPRECATED: This main block is no longer recommended
# Use vectorize_collections() function instead or run via scripts/init_vector_store.py
if __name__ == "__main__":
    import warnings
    warnings.warn(
        "Running vector_store.py directly is DEPRECATED. "
        "Please use: python scripts/init_vector_store.py",
        DeprecationWarning,
        stacklevel=2
    )

    try:
        results = vectorize_collections()

        # Exit with appropriate code
        if results['failed']:
            exit(1)
        elif results['skipped'] and not results['successful']:
            exit(1)
        else:
            exit(0)

    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user (Ctrl+C)")
        exit(130)
    except VectorStoreError as e:
        print(f"ERROR: {e}")
        exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)