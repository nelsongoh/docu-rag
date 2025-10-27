# src/collectors/__init__.py
from .base import BaseDocCollector
from .stripe_collector import StripeDocCollector

"""
Collector registry - add new collectors here
Note that the key in the COLLECTORS dictionary should match the one in the /config/doc_sources.json
"""
COLLECTORS = {
    'stripe': StripeDocCollector,
}

__all__ = ['BaseDocCollector', 'COLLECTORS']