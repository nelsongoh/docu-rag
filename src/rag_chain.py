# src/rag_chain.py
import anthropic
import os
import json
import tiktoken
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from src.vector_store import VectorStoreSingleton
from config.constants import (
    DEFAULT_LLM_CLIENT,
    SUPPORTED_LLM_CLIENTS,
    SUPPORTED_CLAUDE_MODELS,
    DEFAULT_CLAUDE_MODEL,
    RAG_DEFAULT_CONTEXT_SIZE,
    CLAUDE_PRICING,
    TOKENIZER_ENCODING,
)

# Load environment variables
load_dotenv()


class RAGError(Exception):
    """Custom exception for RAG-related errors"""
    pass


class DocumentRAG:
    """
    Document RAG system with configurable client, model, and prompts.

    Features:
    - Configurable LLM client (Anthropic, OpenAI, etc.)
    - Configurable model selection with validation
    - Singleton VectorStore to avoid costly re-initialization
    - Prompt templates loaded from configuration file
    - Error handling for improper initialization
    - Cost estimation before execution
    """

    def __init__(
        self,
        collection_name: str,
        llm_client: str = DEFAULT_LLM_CLIENT,
        model: Optional[str] = None,
        prompt_preset: str = "payment_api",
        chroma_db_path: str = "./chroma_db",
        api_key: Optional[str] = None
    ):
        """
        Initialize DocumentRAG with configurable parameters.

        Args:
            collection_name: Name of the ChromaDB collection to use
            llm_client: LLM client to use ('anthropic' or 'openai')
            model: Model to use. If None, uses DEFAULT_CLAUDE_MODEL
            prompt_preset: Name of the prompt preset from rag_prompts.json
            chroma_db_path: Path to ChromaDB storage directory
            api_key: API key for the LLM client. If None, reads from environment

        Raises:
            RAGError: If initialization parameters are invalid
        """
        # Validate collection_name
        if not collection_name or not isinstance(collection_name, str):
            raise RAGError("collection_name must be a non-empty string")

        # Validate and set LLM client
        if llm_client not in SUPPORTED_LLM_CLIENTS:
            raise RAGError(
                f"Unsupported LLM client: '{llm_client}'. "
                f"Supported clients: {', '.join(SUPPORTED_LLM_CLIENTS.keys())}"
            )

        self.llm_client_type = llm_client
        self.collection_name = collection_name
        self.prompt_preset = prompt_preset

        # Initialize tokenizer for cost estimation
        try:
            self.encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)
        except Exception as e:
            raise RAGError(f"Failed to initialize tokenizer: {e}")

        # Initialize LLM client
        if llm_client == "anthropic":
            self._init_anthropic_client(api_key, model)
        else:
            raise RAGError(f"LLM client '{llm_client}' is not yet implemented")

        # Load prompt templates
        self._load_prompts()

        # Initialize VectorStore singleton
        try:
            self.vector_store = VectorStoreSingleton.get_instance(
                collection_name=collection_name,
                description=f"Collection for {collection_name}",
                metadata={"client": self.llm_client_type, "model": self.model},
                chroma_db_path=chroma_db_path
            )
        except Exception as e:
            raise RAGError(f"Failed to initialize VectorStore: {e}")

    def _init_anthropic_client(self, api_key: Optional[str], model: Optional[str]) -> None:
        """
        Initialize Anthropic client and validate model.

        Args:
            api_key: API key for Anthropic. If None, reads from ANTHROPIC_API_KEY env var
            model: Model name. If None, uses DEFAULT_CLAUDE_MODEL

        Raises:
            RAGError: If API key is missing or model is invalid
        """
        # Get API key
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise RAGError(
                "ANTHROPIC_API_KEY not found. "
                "Please set it in your environment variables or pass it as api_key parameter"
            )

        try:
            self.client = anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            raise RAGError(f"Failed to initialize Anthropic client: {e}")

        # Set and validate model
        if model is None:
            model = DEFAULT_CLAUDE_MODEL

        if model not in SUPPORTED_CLAUDE_MODELS:
            raise RAGError(
                f"Unsupported Claude model: '{model}'. "
                f"Supported models: {', '.join(SUPPORTED_CLAUDE_MODELS.keys())}"
            )

        self.model = SUPPORTED_CLAUDE_MODELS[model]
        self.model_key = model  # Store the key for pricing lookup

    def _load_prompts(self) -> None:
        """
        Load prompt templates from rag_prompts.json configuration file.

        Raises:
            RAGError: If configuration file is missing or invalid
        """
        config_path = Path(__file__).parent.parent / "config" / "rag_prompts.json"

        if not config_path.exists():
            raise RAGError(
                f"Prompt configuration file not found at {config_path}. "
                "Please create config/rag_prompts.json"
            )

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise RAGError(f"Invalid JSON in prompt configuration: {e}")
        except Exception as e:
            raise RAGError(f"Failed to load prompt configuration: {e}")

        # Validate prompt preset exists
        prompts_dict = config.get("prompts", {})
        if self.prompt_preset not in prompts_dict and self.prompt_preset != "default":
            raise RAGError(
                f"Prompt preset '{self.prompt_preset}' not found in configuration. "
                f"Available presets: {', '.join(prompts_dict.keys())}"
            )

        # Load system and user prompts
        if self.prompt_preset in prompts_dict:
            preset = prompts_dict[self.prompt_preset]
            self.system_prompt = preset.get("system_prompt", config.get("default_system_prompt", ""))
            self.user_prompt_template = preset.get("user_prompt_template", config.get("default_user_prompt_template", ""))
        else:
            self.system_prompt = config.get("default_system_prompt", "")
            self.user_prompt_template = config.get("default_user_prompt_template", "")

        if not self.system_prompt or not self.user_prompt_template:
            raise RAGError("System prompt or user prompt template is empty in configuration")

    def retrieve_context(self, query: str, n_results: int = RAG_DEFAULT_CONTEXT_SIZE) -> tuple:
        """
        Retrieve relevant context from vector store.

        Args:
            query: Search query
            n_results: Number of context chunks to retrieve

        Returns:
            Tuple of (formatted_context, sources_list)
        """
        results = self.vector_store.search(query, n_results=n_results)

        # Format context with citations
        context_parts = []
        sources = []

        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            source_id = f"[{i+1}]"
            context_parts.append(f"{source_id} {doc}")
            sources.append({
                'id': source_id,
                'title': metadata.get('title', 'Unknown'),
                'chunk_id': metadata.get('chunk_id', 'Unknown')
            })

        context = "\n\n".join(context_parts)
        return context, sources

    def create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt with retrieved context using loaded template.

        Args:
            query: User's question
            context: Retrieved context from vector store

        Returns:
            Formatted prompt string
        """
        return self.user_prompt_template.format(context=context, query=query)
    
    def estimate_cost(self, query: str, n_results: int = RAG_DEFAULT_CONTEXT_SIZE) -> Dict:
        """
        Estimate the cost of executing a RAG call before actually making it.

        This provides a practical way to understand the cost implications before
        executing the full RAG pipeline.

        Args:
            query: The user's question (for input token estimation)
            n_results: Number of context chunks to retrieve

        Returns:
            Dictionary with cost estimate and token breakdown:
            {
                'input_tokens_estimate': int,
                'output_tokens_estimate': int,
                'input_cost_estimate': float,
                'output_cost_estimate': float,
                'total_cost_estimate': float,
                'model': str,
                'currency': 'USD'
            }

        Raises:
            RAGError: If cost estimation fails or model pricing is unavailable
        """
        try:
            # Retrieve context to estimate size
            context, _ = self.retrieve_context(query, n_results=n_results)

            # Create full prompt to estimate input tokens
            prompt = self.create_prompt(query, context)

            # Count tokens in the prompt
            input_tokens = len(self.encoding.encode(prompt))

            # Estimate output tokens (Claude typically responds with similar length to input)
            # This is a rough estimate; actual output varies
            estimated_output_tokens = min(2000, max(input_tokens // 2, 500))

            # Get pricing for the model
            if self.model_key not in CLAUDE_PRICING:
                raise RAGError(
                    f"Pricing information not available for model '{self.model_key}'. "
                    f"Please update CLAUDE_PRICING in config/constants.py"
                )

            pricing = CLAUDE_PRICING[self.model_key]
            input_price_per_million = pricing["input"]
            output_price_per_million = pricing["output"]

            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * input_price_per_million
            output_cost = (estimated_output_tokens / 1_000_000) * output_price_per_million
            total_cost = input_cost + output_cost

            return {
                'input_tokens_estimate': input_tokens,
                'output_tokens_estimate': estimated_output_tokens,
                'input_cost_estimate': round(input_cost, 6),
                'output_cost_estimate': round(output_cost, 6),
                'total_cost_estimate': round(total_cost, 6),
                'model': self.model_key,
                'currency': 'USD'
            }
        except RAGError:
            raise
        except Exception as e:
            raise RAGError(f"Failed to estimate cost: {e}")

    def generate_response(self, query: str, stream: bool = False) -> Dict:
        """
        Generate response using RAG with configurable parameters.

        Args:
            query: User's question
            stream: Whether to stream the response

        Returns:
            Dictionary with answer, sources, and usage information
        """
        # Step 1: Retrieve relevant context
        context, sources = self.retrieve_context(query)

        # Step 2: Create prompt
        prompt = self.create_prompt(query, context)

        # Step 3: Generate response
        if stream:
            return self._generate_streaming(prompt, sources)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            return {
                'answer': response.content[0].text,
                'sources': sources,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }

    def _generate_streaming(self, prompt: str, sources: List[Dict]):
        """Generate streaming response"""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=2000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                yield text

            # Yield sources at the end
            yield "\n\n---\nSources:\n"
            for source in sources:
                yield f"\n{source['id']} {source['title']}"

# Usage
if __name__ == "__main__":
    try:
        # Initialize RAG system
        # You can customize these parameters:
        # - collection_name: The ChromaDB collection to use
        # - llm_client: 'anthropic' or 'openai'
        # - model: Claude model to use (see SUPPORTED_CLAUDE_MODELS in constants.py)
        # - prompt_preset: Prompt template from rag_prompts.json
        # - api_key: Optional, reads from environment if not provided
        rag = DocumentRAG(
            collection_name="stripe_docs",
            llm_client="anthropic",
            model="claude-sonnet-4",
            prompt_preset="payment_api"
        )

        query = "How do I handle failed payment retries with webhooks?"

        # Estimate cost before executing
        print("Estimating cost...")
        cost_estimate = rag.estimate_cost(query)
        print(f"Cost Estimate for query:")
        print(f"  Input tokens: {cost_estimate['input_tokens_estimate']}")
        print(f"  Output tokens (estimated): {cost_estimate['output_tokens_estimate']}")
        print(f"  Total cost estimate: ${cost_estimate['total_cost_estimate']:.6f} {cost_estimate['currency']}")
        print()

        # Generate response
        print("Generating response...")
        result = rag.generate_response(query)

        print("Answer:")
        print(result['answer'])
        print("\nSources:")
        for source in result['sources']:
            print(f"{source['id']} {source['title']}")
        print(f"\nTokens used: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)