import json
import tiktoken
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path to import config and utils
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from config.constants import (
    MAX_TOKENS_PER_CHUNK,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    TOKENIZER_ENCODING
)
from utils.text_cleaner import TextCleaner

class DocumentChunker:
    def __init__(self, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, max_chunk_size=MAX_TOKENS_PER_CHUNK):
        """
        Initialize document chunker with token-based chunking

        Args:
            chunk_size: Target chunk size in tokens (default: 500)
            chunk_overlap: Overlap between chunks in tokens (default: 50)
            max_chunk_size: Maximum chunk size in tokens (default: 8000)
        """
        if chunk_size > max_chunk_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) cannot exceed max_chunk_size ({max_chunk_size})"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
        self.encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)

        print(f"DocumentChunker initialized:")
        print(f"  Target chunk size: {chunk_size} tokens")
        print(f"  Chunk overlap: {chunk_overlap} tokens")
        print(f"  Maximum chunk size: {max_chunk_size} tokens")
    
    def _clean_content(self, content: str, doc_format: str = 'unknown') -> str:
        """Apply additional cleaning to content before chunking

        This ensures any residual HTML/CSS that made it through collection
        is removed before chunking for optimal semantic retrieval.

        Args:
            content: Raw content from collected documents
            doc_format: Format hint ('html', 'markdown', or 'unknown')

        Returns:
            Cleaned content
        """
        if not content:
            return ""

        # Determine format if not provided
        if doc_format == 'unknown':
            # Heuristic: if content has HTML tags, treat as HTML
            if '<' in content and '>' in content:
                doc_format = 'html' if '<html' in content.lower() or '<body' in content.lower() else 'markdown'
            else:
                doc_format = 'text'

        # Apply format-specific cleaning
        if doc_format == 'html':
            cleaned = TextCleaner.clean_html(content, preserve_code=True)
        elif doc_format == 'markdown':
            cleaned = TextCleaner.clean_markdown(content)
        else:
            cleaned = content

        # Always normalize whitespace
        cleaned = TextCleaner.normalize_whitespace(cleaned)

        return cleaned

    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Chunk documents intelligently

        Strategy:
        1. Clean content to remove residual HTML/CSS
        2. Split by semantic boundaries (sections, paragraphs)
        3. Maintain context with overlap
        4. Keep metadata with each chunk
        """
        chunks = []

        for doc in docs:
            content = doc['content']
            title = doc.get('title', '')
            metadata = doc.get('metadata', {})
            doc_format = doc.get('format', 'unknown')

            # Apply additional cleaning before chunking
            content = self._clean_content(content, doc_format)
            
            # Split into paragraphs first
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = len(self.encoding.encode(para))

                # If paragraph itself is too large, split it further
                if para_tokens > self.max_chunk_size:
                    print(f"  WARNING: Paragraph with {para_tokens} tokens exceeds max limit, splitting...")
                    # Split oversized paragraph by sentences
                    sub_chunks = self._split_oversized_text(para, title, metadata, len(chunks))
                    chunks.extend(sub_chunks)
                    continue

                # If adding this paragraph exceeds chunk size, save current chunk
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Validate chunk size before adding
                    if current_tokens > self.max_chunk_size:
                        print(f"  WARNING: Chunk with {current_tokens} tokens exceeds max limit, truncating...")
                        current_chunk = self._truncate_to_tokens(current_chunk, self.max_chunk_size)

                    chunks.append(self._create_chunk(
                        current_chunk,
                        title,
                        metadata,
                        len(chunks)
                    ))

                    # Start new chunk with overlap
                    # Keep last sentence(s) for context
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + para
                    current_tokens = len(self.encoding.encode(current_chunk))
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens
            
            # Add final chunk
            if current_chunk:
                chunks.append(self._create_chunk(
                    current_chunk, 
                    title, 
                    metadata, 
                    len(chunks)
                ))
        
        return chunks
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
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

    def _split_oversized_text(self, text: str, title: str, metadata: Dict, start_chunk_id: int) -> List[Dict]:
        """Split oversized text into multiple chunks

        Args:
            text: Text to split
            title: Document title
            metadata: Document metadata
            start_chunk_id: Starting chunk ID

        Returns:
            List of chunks
        """
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence + '. '))

            # If single sentence exceeds max, truncate it
            if sentence_tokens > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        title,
                        metadata,
                        start_chunk_id + len(chunks)
                    ))
                    current_chunk = ""
                    current_tokens = 0

                # Truncate the oversized sentence
                truncated = self._truncate_to_tokens(sentence, self.max_chunk_size)
                chunks.append(self._create_chunk(
                    truncated,
                    title,
                    metadata,
                    start_chunk_id + len(chunks)
                ))
                continue

            # If adding sentence exceeds max, start new chunk
            if current_tokens + sentence_tokens > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        title,
                        metadata,
                        start_chunk_id + len(chunks)
                    ))
                current_chunk = sentence + '. '
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence + '. '
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                title,
                metadata,
                start_chunk_id + len(chunks)
            ))

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get last few sentences for overlap"""
        sentences = text.split('. ')
        overlap_sentences = sentences[-2:] if len(sentences) > 2 else sentences
        overlap_text = '. '.join(overlap_sentences)

        # Ensure overlap doesn't exceed max size
        overlap_tokens = len(self.encoding.encode(overlap_text))
        if overlap_tokens > self.chunk_overlap:
            overlap_text = self._truncate_to_tokens(overlap_text, self.chunk_overlap)

        return overlap_text

    def _create_chunk(self, content: str, title: str, metadata: Dict, chunk_id: int) -> Dict:
        """Create chunk with metadata"""
        return {
            'chunk_id': chunk_id,
            'content': content.strip(),
            'title': title,
            'metadata': {
                **metadata,
                'chunk_id': chunk_id,
                'char_count': len(content),
                'token_count': len(self.encoding.encode(content))
            }
        }
    
    def process_and_save(self, input_path: str, output_path: str):
        """Load, chunk, and save documents"""
        with open(input_path, 'r') as f:
            docs = json.load(f)
        
        chunks = self.chunk_documents(docs)
        
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f"Created {len(chunks)} chunks from {len(docs)} documents")
        return chunks

    def process_all_raw_files(self, raw_dir: str, processed_dir: str):
        """Process all JSON files in the raw directory

        Args:
            raw_dir: Path to directory containing raw JSON files
            processed_dir: Path to directory where chunked files will be saved
        """
        raw_path = Path(raw_dir)
        processed_path = Path(processed_dir)

        # Create processed directory if it doesn't exist
        processed_path.mkdir(parents=True, exist_ok=True)

        # Find all JSON files in raw directory
        json_files = list(raw_path.glob('*.json'))

        if not json_files:
            print(f"No JSON files found in {raw_dir}")
            return

        print(f"Found {len(json_files)} file(s) to process")

        for input_file in json_files:
            # Generate output filename with chunked_ prefix
            output_filename = f"chunked_{input_file.name}"
            output_file = processed_path / output_filename

            print(f"\nProcessing: {input_file.name}")

            try:
                # Load documents
                with open(input_file, 'r') as f:
                    docs = json.load(f)

                # Chunk documents
                chunks = self.chunk_documents(docs)

                # Save chunks
                with open(output_file, 'w') as f:
                    json.dump(chunks, f, indent=2)

                print(f"  ✓ Created {len(chunks)} chunks from {len(docs)} documents")
                print(f"  ✓ Saved to: {output_filename}")

            except Exception as e:
                print(f"  ✗ Error processing {input_file.name}: {e}")
                continue

        print(f"\nProcessing complete!")


# Usage
if __name__ == "__main__":
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

    # Process all files in data/raw directory
    chunker.process_all_raw_files(
        raw_dir='data/raw',
        processed_dir='data/processed'
    )