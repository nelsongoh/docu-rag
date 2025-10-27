# src/document_manager.py
"""
Document and collection management utilities for the RAG system.

Provides helpers to load available documentation sources and their
corresponding vector store collections.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class DocumentManager:
    """Manage available documentation sources and their collections."""

    def __init__(self, config_path: str = "config/doc_sources.json", vector_store_config_path: str = "config/vector_store_config.json"):
        """
        Initialize DocumentManager.

        Args:
            config_path: Path to doc_sources.json configuration
            vector_store_config_path: Path to vector_store_config.json
        """
        self.config_path = Path(config_path)
        self.vector_store_config_path = Path(vector_store_config_path)
        self._doc_sources = None
        self._vector_store_config = None

    def load_doc_sources(self) -> Dict:
        """
        Load and cache documentation sources configuration.

        Returns:
            Dictionary with sources configuration
        """
        if self._doc_sources is not None:
            return self._doc_sources

        if not self.config_path.exists():
            raise FileNotFoundError(f"Document sources config not found at {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                self._doc_sources = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}")

        return self._doc_sources

    def load_vector_store_config(self) -> Dict:
        """
        Load and cache vector store configuration.

        Returns:
            Dictionary with vector store configuration
        """
        if self._vector_store_config is not None:
            return self._vector_store_config

        if not self.vector_store_config_path.exists():
            raise FileNotFoundError(f"Vector store config not found at {self.vector_store_config_path}")

        try:
            with open(self.vector_store_config_path, 'r') as f:
                self._vector_store_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.vector_store_config_path}: {e}")

        return self._vector_store_config

    def get_enabled_sources(self) -> List[Dict]:
        """
        Get list of enabled documentation sources.

        Returns:
            List of enabled source configurations
        """
        config = self.load_doc_sources()
        sources = config.get('sources', [])
        return [s for s in sources if s.get('enabled', False)]

    def get_source_by_name(self, name: str) -> Optional[Dict]:
        """
        Get a documentation source by name.

        Args:
            name: Source name

        Returns:
            Source configuration or None if not found
        """
        config = self.load_doc_sources()
        sources = config.get('sources', [])
        for source in sources:
            if source.get('name') == name:
                return source
        return None

    def get_collection_names(self) -> List[str]:
        """
        Get list of all available collection names from vector store config.

        Returns:
            List of collection names
        """
        config = self.load_vector_store_config()
        collections = config.get('collections', [])
        return [c['name'] for c in collections]

    def get_collection_by_name(self, name: str) -> Optional[Dict]:
        """
        Get collection configuration by name.

        Args:
            name: Collection name

        Returns:
            Collection configuration or None if not found
        """
        config = self.load_vector_store_config()
        collections = config.get('collections', [])
        for collection in collections:
            if collection.get('name') == name:
                return collection
        return None

    def get_source_to_collection_mapping(self) -> Dict[str, str]:
        """
        Create a mapping of source names to collection names.

        Returns:
            Dictionary mapping source names to collection names
        """
        mapping = {}
        vector_config = self.load_vector_store_config()
        collections = vector_config.get('collections', [])

        for collection in collections:
            source_name = collection.get('source')
            if source_name:
                mapping[source_name] = collection.get('name')

        return mapping

    def get_documentation_options(self) -> Dict[str, str]:
        """
        Get user-friendly documentation options for display.

        Returns:
            Dictionary mapping display names to collection names
        """
        options = {}
        vector_config = self.load_vector_store_config()
        collections = vector_config.get('collections', [])

        for collection in collections:
            name = collection.get('name')
            description = collection.get('description', name)
            options[description] = name

        return options