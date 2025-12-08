"""
LangChain integration layer for LLM trading system.

This module provides:
- LLMFactory: Factory for creating ChatOpenAI instances with custom base_url
- Trading tools: LangChain @tool decorated functions for trading actions
- Context adapters: Bridge between existing ContextBuilder and LangChain
"""

from .llm_factory import LLMFactory

__all__ = ["LLMFactory"]
