# src/collectors/stripe_collector.py
import re
import requests
import sys
from pathlib import Path
from typing import Dict, Any
from .base import BaseDocCollector

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_cleaner import TextCleaner


class StripeDocCollector(BaseDocCollector):
    """Collector for Stripe documentation (supports both HTML and Markdown)"""

    def __init__(self):
        super().__init__(source_name='stripe', base_url='https://docs.stripe.com')

    def _parse_content(self, url: str, response: requests.Response) -> Dict[str, Any]:
        """Parse Stripe documentation from response

        Supports both HTML pages and Markdown files.
        """
        content_type = response.headers.get('Content-Type', '').lower()

        # Check if it's markdown by content-type or URL extension
        is_markdown = (
            'markdown' in content_type or
            'text/plain' in content_type or
            url.endswith('.md')
        )

        if is_markdown:
            return self._parse_markdown(url, response.text)
        else:
            return self._parse_html(url, response.text)

    def _parse_markdown(self, url: str, content: str) -> Dict[str, Any]:
        """Parse Markdown content with cleaning"""
        if not content or not content.strip():
            return None

        # Clean markdown (remove embedded HTML, etc.)
        cleaned_content = TextCleaner.clean_markdown(content)
        cleaned_content = TextCleaner.normalize_whitespace(cleaned_content)

        # Extract title from first H1 heading in markdown
        title_match = re.search(r'^#\s+(.+)$', cleaned_content, re.MULTILINE)
        title = title_match.group(1) if title_match else ''

        return {
            'url': url,
            'title': title,
            'content': cleaned_content,
            'format': 'markdown',
            'metadata': {
                'source': self.source_name,
                'section': self._extract_section(url)
            }
        }

    def _parse_html(self, url: str, html_content: str) -> Dict[str, Any]:
        """Parse HTML content with thorough cleaning"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find main content
        content = soup.find('article') or soup.find('main')
        if not content:
            return None

        # Convert to string for cleaning
        content_html = str(content)

        # Clean HTML thoroughly
        cleaned_content = TextCleaner.clean_html(content_html, preserve_code=True)
        cleaned_content = TextCleaner.normalize_whitespace(cleaned_content)

        # Extract title
        title = soup.find('h1').text if soup.find('h1') else ''

        return {
            'url': url,
            'title': title,
            'content': cleaned_content,
            'format': 'html',
            'metadata': {
                'source': self.source_name,
                'section': self._extract_section(url)
            }
        }

    def _extract_section(self, url: str) -> str:
        """Extract section from Stripe URL"""
        # Remove .md extension if present
        url_path = url.replace('.md', '')
        parts = url_path.replace('https://stripe.com/docs/', '').replace('https://docs.stripe.com/', '').split('/')
        return parts[0] if parts else 'general'