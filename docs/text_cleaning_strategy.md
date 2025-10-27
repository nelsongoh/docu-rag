# Text Cleaning Strategy for HTML and CSS Markup

## Overview
The documentation collection and processing pipeline implements a **two-stage text cleaning strategy** to optimize token usage while preserving semantic meaning, for documents with HTML and CSS markup.

## Problem Statement

Documentation sources often contain:
- HTML tags (`<div>`, `<span>`, etc.)
- CSS styles (inline and `<style>` blocks)
- JavaScript code (`<script>` tags)
- Navigation elements, headers, footers
- Excessive whitespace and formatting
- Embedded HTML in Markdown files

These elements consume tokens without adding value for RAG (Retrieval-Augmented Generation) systems.

## Solution: Two-Stage Cleaning

### Stage 1: Collection Time (in `data_collector.py`)

**Location**: `src/utils/text_cleaner.py`
**When**: During web scraping and content extraction
**Purpose**: Remove structural bloat while preserving semantic content

**Operations:**
1. **Remove unwanted HTML elements**
   - `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`
   - HTML comments

2. **Clean embedded HTML from Markdown**
   - Remove `<script>` and `<style>` tags
   - Strip HTML tags while preserving text
   - Protect code blocks during cleaning

3. **Preserve important structure**
   - Code blocks (marked with `[CODE]...[/CODE]`)
   - Paragraph breaks
   - List structures

4. **Basic normalization**
   - Convert HTML to plain text
   - Initial whitespace cleanup

### Stage 2: Processing Time (in `data_processor.py`)

**When**: During document chunking
**Purpose**: Final normalization for optimal token efficiency

**Operations:**
1. **Whitespace normalization**
   - Multiple spaces → single space
   - Excessive newlines → max 2 newlines
   - Remove trailing whitespace

2. **Code marker handling**
   - Keep markers for chunking decisions
   - Option to remove in aggressive mode

## Implementation

### TextCleaner Class

```python
from utils.text_cleaner import TextCleaner

# Clean HTML content
cleaned = TextCleaner.clean_html(html_content, preserve_code=True)

# Clean Markdown content
cleaned = TextCleaner.clean_markdown(markdown_content)

# Normalize whitespace
normalized = TextCleaner.normalize_whitespace(text)

# All-in-one cleaning
cleaned = TextCleaner.clean_text(content, format='html', aggressive=False)
```

### Convenience Functions

```python
from utils import clean_html_content, clean_markdown_content, normalize_text

# Quick HTML cleaning
clean_html = clean_html_content(raw_html)

# Quick Markdown cleaning
clean_md = clean_markdown_content(raw_markdown)

# Quick normalization
normalized = normalize_text(text)
```

## Example: Before and After

### Before Cleaning (HTML):
```html
<article>
  <script>analytics.track();</script>
  <style>.button { color: blue; }</style>
  <div class="content">
    <h1>API Documentation</h1>
    <p>  This   is   a   paragraph  </p>


    <pre><code>function example() {
      return true;
    }</code></pre>
  </div>
  <nav>Navigation links...</nav>
</article>
```

### After Cleaning:
```
API Documentation

This is a paragraph

[CODE]
function example() {
  return true;
}
[/CODE]
```

## Configuration Options

### For Collectors (`stripe_collector.py`)

```python
# Preserve code blocks (default)
TextCleaner.clean_html(content, preserve_code=True)

# Remove code blocks (if not needed)
TextCleaner.clean_html(content, preserve_code=False)
```

### For Processing

```python
# Standard cleaning (keeps code markers)
TextCleaner.clean_text(content, aggressive=False)

# Aggressive cleaning (removes code markers)
TextCleaner.clean_text(content, aggressive=True)
```

## Benefits

1. **Token Efficiency**: Reduced token counts.
2. **Cost Savings**: Fewer tokens = lower embedding costs
3. **Better Retrieval**: Less noise = more relevant matches
4. **Preserved Semantics**: Code blocks and structure maintained
5. **Flexibility**: Two-stage approach allows re-processing
6. **Debugging**: Raw files retain some structure

## Usage in Pipeline

### 1. Data Collection
```bash
python src/data_collector.py
```
- Fetches documentation
- Applies TextCleaner during parsing
- Saves cleaned content to `data/raw/`

### 2. Data Processing
```bash
python src/data_processor.py
```
- Loads cleaned content
- Applies final normalization
- Chunks with token awareness
- Saves to `data/processed/`

### 3. Vector Store Creation
```bash
python src/vector_store.py
```
- Loads processed chunks
- Validates token limits (8000 max)
- Embeddings created from clean text

## Adding Cleaning to New Collectors

When creating a new collector, integrate TextCleaner:

```python
from utils.text_cleaner import TextCleaner

class MyAPIDocCollector(BaseDocCollector):
    def _parse_html(self, url: str, html: str) -> Dict:
        # Extract main content
        soup = BeautifulSoup(html, 'html.parser')
        content = soup.find('main')

        # Clean HTML
        cleaned = TextCleaner.clean_html(str(content))
        cleaned = TextCleaner.normalize_whitespace(cleaned)

        return {
            'content': cleaned,
            # ... other fields
        }
```

## Future Enhancements

Potential improvements:
1. Language-specific code block preservation
2. Table structure preservation
3. Link text extraction options
4. Custom element removal rules per collector
5. Caching of cleaned content

## Summary

The two-stage text cleaning strategy provides:
- ✓ Token savings
- ✓ Preserved semantic meaning
- ✓ Clean, debuggable raw files
- ✓ Flexible re-processing
- ✓ Minimal performance impact

This approach balances efficiency, quality, and flexibility for production RAG systems.
