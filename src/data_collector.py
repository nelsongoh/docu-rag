# src/data_collector.py
import json
from pathlib import Path
from typing import Dict, Any
from collectors import BaseDocCollector, COLLECTORS


class CollectorNotFoundError(Exception):
    """Raised when a requested collector type is not registered"""
    pass


class ConfigurationError(Exception):
    """Raised when configuration file is missing or invalid"""
    pass


class DocCollectorFactory:
    """Factory for creating documentation collectors"""

    @classmethod
    def create_collector(cls, source_config: Dict[str, Any]) -> BaseDocCollector:
        """Create a collector based on configuration

        Args:
            source_config: Configuration dictionary containing 'name' key

        Returns:
            Instance of the appropriate collector

        Raises:
            CollectorNotFoundError: If collector type is not registered
        """
        collector_type = source_config.get('name', '').lower()

        if collector_type not in COLLECTORS:
            available = ', '.join(COLLECTORS.keys())
            raise CollectorNotFoundError(
                f"Collector '{collector_type}' does not exist. "
                f"Available collectors: {available}"
            )

        return COLLECTORS[collector_type]()

    @classmethod
    def list_collectors(cls):
        """List all registered collector types"""
        return list(COLLECTORS.keys())


# Usage
if __name__ == "__main__":
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'doc_sources.json'

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found at: {config_path}\n"
            f"Please create the configuration file with your documentation sources."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'sources' not in config:
        raise ConfigurationError(
            "Invalid configuration: 'sources' key not found in config file."
        )

    # Process each enabled source
    for source in config['sources']:
        if not source.get('enabled', True):
            continue

        print(f"Collecting documentation from {source['name']}...")

        try:
            # Create collector
            collector = DocCollectorFactory.create_collector(source)

            # Collect documents
            collector.collect_doc_pages(source['urls'])

            # Save documents
            output_path = Path(__file__).parent.parent / source['output_file']
            output_path.parent.mkdir(parents=True, exist_ok=True)
            collector.save_docs(str(output_path))

            print(f"Saved {len(collector.docs)} documents to {source['output_file']}")

        except CollectorNotFoundError as e:
            print(f"ERROR: {e}")
            print(f"Please create a custom collector for '{source['name']}' or use an existing one.")
            continue
        except Exception as e:
            print(f"ERROR: Failed to collect from {source['name']}: {e}")
            continue