"""
Execution nodes for LangGraph.

Provides:
- ExecutorAgentNode: NEW LLM-based final decision maker
- ValidatorNode: (Legacy) Validates trading decisions against risk rules
- ExecutorNode: (Legacy) Executes approved trading actions via tools
"""

from .executor_agent import executor_agent_node
from .validator_node import validator_node  # Legacy
from .executor_node import executor_node, format_execution_summary  # Legacy

__all__ = [
    "executor_agent_node",
    "validator_node",
    "executor_node",
    "format_execution_summary",
]
