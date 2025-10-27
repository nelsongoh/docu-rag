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

# RAG Configuration
# Supported LLM clients
SUPPORTED_LLM_CLIENTS = {
    "anthropic": "anthropic",
    "openai": "openai"
}

# Default LLM client
DEFAULT_LLM_CLIENT = "anthropic"

# Supported Claude models
SUPPORTED_CLAUDE_MODELS = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
}

# Default Claude model
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4"

# RAG Generation parameters
RAG_MAX_TOKENS = 2000
RAG_TEMPERATURE = 0.3  # Lower temperature for more factual responses
RAG_DEFAULT_CONTEXT_SIZE = 5  # Default number of context chunks to retrieve

# Model pricing for cost estimation (per 1M input/output tokens)
# Last updated: 2025-01-24
CLAUDE_PRICING = {
    "claude-sonnet-4": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-5-sonnet": {
        "input": 3.00,
        "output": 15.00
    },
    "claude-3-5-haiku": {
        "input": 0.80,
        "output": 4.00
    }
}