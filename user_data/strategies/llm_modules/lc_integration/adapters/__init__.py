"""
Adapters for integrating existing modules with LangChain.

Provides:
- ContextAdapter: Bridges ContextBuilder output to LangChain message format
"""

from .context_adapter import ContextAdapter

__all__ = ["ContextAdapter"]
