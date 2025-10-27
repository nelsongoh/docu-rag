# src/collectors/base.py
import requests
from bs4 import BeautifulSoup
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseDocCollector(ABC):
    """Abstract base class for documentation collectors"""

    def __init__(self, source_name: str, base_url: str):
        self.source_name = source_name
        self.base_url = base_url
        self.docs = []

    def collect_doc_pages(self, urls: List[str]):
        """Collect content from documentation pages"""
        for idx, url in enumerate(urls):
            try:
                print(f"Fetching URL {idx+1} of {len(urls)}...")
                response = requests.get(url)
                response.raise_for_status()

                # Extract content using source-specific logic
                doc = self._parse_content(url, response)
                if doc:
                    self.docs.append(doc)

                time.sleep(1)  # Be respectful with rate limiting

            except Exception as e:
                print(f"Error collecting {url}: {e}")

    @abstractmethod
    def _parse_content(self, url: str, response: requests.Response) -> Dict[str, Any]:
        """Parse content from response - must be implemented by subclasses

        Args:
            url: The URL being processed
            response: The requests Response object containing the fetched content

        Returns:
            Dictionary with parsed document data, or None to skip
        """
        pass

    @abstractmethod
    def _extract_section(self, url: str) -> str:
        """Extract section from URL - must be implemented by subclasses"""
        pass

    def save_docs(self, filepath: str):
        """Save collected docs to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.docs, f, indent=2)

    def get_docs(self) -> List[Dict[str, Any]]:
        """Return collected documents"""
        return self.docs