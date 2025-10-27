# src/utils/text_cleaner.py
"""
Text cleaning utilities for documentation content.
Removes HTML/CSS markup and normalizes text to optimize token usage.
"""

import re
from typing import Optional
from bs4 import BeautifulSoup, NavigableString, Tag


class TextCleaner:
    """Clean and normalize text content from documentation"""

    @staticmethod
    def clean_html(html_content: str, preserve_code: bool = True) -> str:
        """Clean HTML content and extract meaningful text

        Args:
            html_content: Raw HTML content
            preserve_code: Whether to preserve code blocks with markers

        Returns:
            Cleaned text content
        """
        if not html_content or not html_content.strip():
            return ""

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove unwanted elements (scripts, styles, navigation, etc.)
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside',
                                       'noscript', 'iframe', 'svg', 'canvas']):
            element.decompose()

        # Remove elements with non-semantic roles (ads, social media, etc.)
        for element in soup.find_all(class_=re.compile(r'(ad|advertisement|social|share|comment|sidebar|widget)', re.I)):
            element.decompose()

        # Remove CSS style attributes and non-semantic attributes from all elements
        for element in soup.find_all(True):  # True finds all tags
            # Remove style attributes
            if element.has_attr('style'):
                del element['style']
            # Remove class attributes (not needed for semantic content)
            if element.has_attr('class'):
                del element['class']
            # Remove id attributes (not needed for semantic content)
            if element.has_attr('id'):
                del element['id']
            # Remove data attributes
            attrs_to_remove = [attr for attr in element.attrs if attr.startswith('data-')]
            for attr in attrs_to_remove:
                del element[attr]
            # Remove event handlers (onclick, onload, etc.)
            event_attrs = [attr for attr in element.attrs if attr.startswith('on')]
            for attr in event_attrs:
                del element[attr]

        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()

        # Handle code blocks specially if preserving
        if preserve_code:
            for code in soup.find_all(['code', 'pre']):
                # Mark code blocks with special markers
                code.insert_before('\n[CODE]\n')
                code.insert_after('\n[/CODE]\n')

        # Extract text with structure
        text = soup.get_text(separator='\n', strip=True)

        return text

    @staticmethod
    def clean_markdown(markdown_content: str) -> str:
        """Clean markdown content that may contain embedded HTML and CSS

        Args:
            markdown_content: Raw markdown content

        Returns:
            Cleaned markdown
        """
        if not markdown_content or not markdown_content.strip():
            return ""

        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', markdown_content, flags=re.DOTALL)

        # Remove HTML tags (but preserve code blocks)
        # First, protect code blocks
        code_blocks = []
        def save_code(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        # Protect fenced code blocks
        content = re.sub(r'```[\s\S]*?```', save_code, content)
        # Protect inline code
        content = re.sub(r'`[^`]+`', save_code, content)

        # Remove script tags and their content
        content = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', content, flags=re.IGNORECASE)

        # Remove style tags and their content (CSS)
        content = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', content, flags=re.IGNORECASE)

        # Remove inline CSS style attributes from HTML tags
        content = re.sub(r'\s+style=["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)

        # Remove class attributes from HTML tags
        content = re.sub(r'\s+class=["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)

        # Remove id attributes from HTML tags
        content = re.sub(r'\s+id=["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)

        # Remove data- attributes from HTML tags
        content = re.sub(r'\s+data-[a-z-]+=["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)

        # Remove event handler attributes (onclick, onload, etc.)
        content = re.sub(r'\s+on[a-z]+=["\'][^"\']*["\']', '', content, flags=re.IGNORECASE)

        # Remove non-semantic HTML tags (keep structure: div, span, etc. will be removed with content preserved)
        # Remove these tags but keep their content
        for tag in ['div', 'span', 'section', 'article', 'nav', 'aside', 'header', 'footer', 'main']:
            content = re.sub(f'<{tag}[^>]*>', '', content, flags=re.IGNORECASE)
            content = re.sub(f'</{tag}>', '', content, flags=re.IGNORECASE)

        # Remove these tags and their content entirely
        for tag in ['iframe', 'noscript', 'svg', 'canvas', 'form', 'input', 'button']:
            content = re.sub(f'<{tag}[^>]*>[\s\S]*?</{tag}>', '', content, flags=re.IGNORECASE)
            # Also handle self-closing tags
            content = re.sub(f'<{tag}[^>]*/?>', '', content, flags=re.IGNORECASE)

        # Remove any remaining HTML tags (preserving content)
        content = re.sub(r'<[^>]+>', '', content)

        # Restore code blocks
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f"__CODE_BLOCK_{i}__", code_block)

        return content

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace to reduce token usage

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Replace multiple newlines with max 2 newlines (preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def remove_code_markers(text: str) -> str:
        """Remove code block markers added during HTML processing

        Args:
            text: Text with code markers

        Returns:
            Text without markers
        """
        text = re.sub(r'\[CODE\]\s*', '', text)
        text = re.sub(r'\s*\[/CODE\]', '', text)
        return text

    @staticmethod
    def clean_text(text: str, format: str = 'html', aggressive: bool = False) -> str:
        """Main entry point for text cleaning

        Args:
            text: Raw text content
            format: Format of content ('html', 'markdown', or 'text')
            aggressive: Whether to apply aggressive cleanup (removes code markers)

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Stage 1: Format-specific cleaning
        if format == 'html':
            cleaned = TextCleaner.clean_html(text, preserve_code=not aggressive)
        elif format == 'markdown':
            cleaned = TextCleaner.clean_markdown(text)
        else:
            cleaned = text

        # Stage 2: Normalize whitespace
        cleaned = TextCleaner.normalize_whitespace(cleaned)

        # Stage 3: Remove code markers if aggressive
        if aggressive:
            cleaned = TextCleaner.remove_code_markers(cleaned)

        return cleaned


# Convenience functions
def clean_html_content(html: str) -> str:
    """Clean HTML content - convenience wrapper"""
    return TextCleaner.clean_text(html, format='html', aggressive=False)


def clean_markdown_content(markdown: str) -> str:
    """Clean markdown content - convenience wrapper"""
    return TextCleaner.clean_text(markdown, format='markdown', aggressive=False)


def normalize_text(text: str) -> str:
    """Normalize text whitespace - convenience wrapper"""
    return TextCleaner.normalize_whitespace(text)