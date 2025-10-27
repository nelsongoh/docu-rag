# config/constants.py
"""
Central configuration constants for the docu-rag application
"""

# Token limits (aligned with OpenAI embedding API limits)
# The actual OpenAI limit is ~8191 tokens for text-embedding-3-small/large
# We use 8000 as a safe margin to account for encoding differences
MAX_TOKENS_PER_CHUNK = 8000

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500  # Target chunk size in tokens
DEFAULT_CHUNK_OVERLAP = 50  # Overlap between chunks in tokens

# OpenAI embedding models
VALID_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
}

# Embedding model pricing (per 1M tokens)
# Last updated: 2025-01-24
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,  # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13,  # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.10,  # $0.10 per 1M tokens (legacy)
}

PRICING_DATE = "2025-01-24"

# Tokenizer encoding
TOKENIZER_ENCODING = "cl100k_base"  # Used by GPT-3.5, GPT-4, and text-embedding-3-* models