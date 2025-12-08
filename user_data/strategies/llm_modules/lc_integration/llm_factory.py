"""
LLM Factory for creating LangChain ChatOpenAI instances.

Provides factory methods to create LLM instances with:
- Custom base_url (OpenAI-compatible API endpoints)
- Configurable temperature, max_tokens, timeout
- Model-specific configurations (e.g., Gemini thinking mode)
- Retry policies for resilience
"""

import logging
from typing import Dict, Any, Optional, List, Union
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory for creating LangChain ChatOpenAI instances.

    Supports any OpenAI-compatible API endpoint via base_url configuration.
    Preserves existing llm_config patterns from the trading system.
    """

    # Default configurations
    DEFAULT_CONFIG = {
        "api_base": "http://host.docker.internal:3120",
        "api_key": "not-needed",
        "model": "gemini-flash-lite-latest",
        "temperature": 0.0,
        "max_tokens": 65536,
        "timeout": 60,
        "retry_times": 3,
    }

    # Model-specific temperature recommendations
    MODEL_TEMPERATURES = {
        # Analysis tasks (deterministic)
        "indicator": 0.1,
        "trend": 0.1,
        "sentiment": 0.1,
        "pattern": 0.1,
        "aggregator": 0.0,
        # Debate tasks (slightly creative for arguments)
        "bull": 0.2,
        "bear": 0.2,
        # Judge (deterministic for fair verdict)
        "judge": 0.0,
        # Default
        "default": 0.0,
    }

    @classmethod
    def create_chat_model(
        cls,
        llm_config: Dict[str, Any],
        task_type: str = "default",
        **overrides
    ) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance with custom configuration.

        Args:
            llm_config: LLM configuration dict from config.json
            task_type: Type of task for temperature recommendation
                      ("indicator", "trend", "bull", "bear", "judge", etc.)
            **overrides: Additional parameters to override config

        Returns:
            Configured ChatOpenAI instance

        Example:
            llm = LLMFactory.create_chat_model(
                config["llm_config"],
                task_type="bull",
                temperature=0.3  # Override recommended temperature
            )
        """
        # Merge defaults with provided config
        config = {**cls.DEFAULT_CONFIG, **llm_config}

        # Determine temperature (priority: override > task_type > config > default)
        if "temperature" in overrides:
            temperature = overrides["temperature"]
        elif task_type in cls.MODEL_TEMPERATURES:
            temperature = cls.MODEL_TEMPERATURES[task_type]
        else:
            temperature = config.get("temperature", cls.DEFAULT_CONFIG["temperature"])

        # Extract configuration
        api_base = config.get("api_base", cls.DEFAULT_CONFIG["api_base"])
        api_key = config.get("api_key", cls.DEFAULT_CONFIG["api_key"])
        model = config.get("model", cls.DEFAULT_CONFIG["model"])
        max_tokens = overrides.get("max_tokens", config.get("max_tokens", cls.DEFAULT_CONFIG["max_tokens"]))
        timeout = overrides.get("timeout", config.get("timeout", cls.DEFAULT_CONFIG["timeout"]))

        # model_kwargs for additional standard OpenAI API parameters (must be dict, not None)
        model_kwargs = {}

        # Build extra_body for provider-specific custom parameters
        # NOTE: extra_body must be passed explicitly, NOT via model_kwargs
        # See: https://python.langchain.com/docs/integrations/chat/openai/
        extra_body = None

        # Gemini model special configuration: enable thinking mode
        if model.startswith("gemini-"):
            extra_body = {
                "google": {
                    "thinking_config": {
                        "thinking_budget": 24576,
                        "include_thoughts": True
                    }
                }
            }

        # Create ChatOpenAI instance
        chat_model = ChatOpenAI(
            base_url=api_base,
            api_key=api_key or "not-needed",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            model_kwargs=model_kwargs,  # For standard OpenAI API params
            extra_body=extra_body,  # For provider-specific custom params (e.g., Google Gemini)
        )

        # logger.info(
        #     f"Created ChatOpenAI: base_url={api_base}, model={model}, "
        #     f"temp={temperature}, task={task_type}"
        # )

        return chat_model

    @classmethod
    def create_vision_model(
        cls,
        llm_config: Dict[str, Any],
        **overrides
    ) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance optimized for vision tasks.

        Vision tasks may require different configurations:
        - Gemini: disable thinking_budget for vision (conflicts with image input)

        Args:
            llm_config: LLM configuration dict
            **overrides: Additional parameters to override

        Returns:
            Configured ChatOpenAI for vision tasks
        """
        config = {**cls.DEFAULT_CONFIG, **llm_config}

        api_base = config.get("api_base", cls.DEFAULT_CONFIG["api_base"])
        api_key = config.get("api_key", cls.DEFAULT_CONFIG["api_key"])
        model = config.get("model", cls.DEFAULT_CONFIG["model"])
        temperature = overrides.get("temperature", config.get("temperature", 0.1))
        max_tokens = overrides.get("max_tokens", config.get("max_tokens"))
        timeout = overrides.get("timeout", config.get("timeout", 90))  # Longer for vision

        # Vision calls: no thinking_budget (conflicts with image input)
        model_kwargs = {}

        chat_model = ChatOpenAI(
            base_url=api_base,
            api_key=api_key or "not-needed",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            model_kwargs=model_kwargs,  # Always pass dict, never None
        )

        # logger.debug(f"Created Vision ChatOpenAI: model={model}, temp={temperature}")

        return chat_model

    @classmethod
    def create_with_tools(
        cls,
        llm_config: Dict[str, Any],
        tools: List[Any],
        tool_choice: str = "required",
        task_type: str = "default",
        **overrides
    ) -> BaseChatModel:
        """
        Create a ChatOpenAI instance bound with tools.

        Args:
            llm_config: LLM configuration dict
            tools: List of LangChain tools to bind
            tool_choice: Tool choice strategy ("auto", "required", "none")
            task_type: Type of task for temperature
            **overrides: Additional parameters

        Returns:
            ChatOpenAI bound with tools
        """
        chat_model = cls.create_chat_model(llm_config, task_type, **overrides)

        if tools:
            return chat_model.bind_tools(tools, tool_choice=tool_choice)

        return chat_model

    @classmethod
    def get_retry_config(cls, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get retry configuration for LangGraph RetryPolicy.

        Args:
            llm_config: LLM configuration dict

        Returns:
            Dict with retry configuration for RetryPolicy
        """
        config = {**cls.DEFAULT_CONFIG, **llm_config}

        return {
            "max_attempts": config.get("retry_times", 3),
            "initial_interval": 1.0,
            "max_interval": 60.0,
            "backoff_multiplier": 2.0,
        }


# Convenience function for simple usage
def create_llm(
    llm_config: Dict[str, Any],
    task_type: str = "default",
    **kwargs
) -> ChatOpenAI:
    """
    Convenience function to create an LLM instance.

    Args:
        llm_config: LLM configuration dict
        task_type: Task type for temperature recommendation
        **kwargs: Additional overrides

    Returns:
        Configured ChatOpenAI instance
    """
    return LLMFactory.create_chat_model(llm_config, task_type, **kwargs)


class VisionModelWithRetry:
    """
    Vision model wrapper with retry logic.

    Wraps a ChatOpenAI model to add automatic retry with exponential backoff
    for vision calls, which are more prone to transient failures.
    """

    def __init__(self, model: ChatOpenAI, max_retries: int = 3):
        """
        Initialize the retry wrapper.

        Args:
            model: The ChatOpenAI model to wrap
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.model = model
        self.max_retries = max_retries

        # Create retry-wrapped invoke method if tenacity is available
        if TENACITY_AVAILABLE:
            self._invoke_with_retry = retry(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                reraise=True,
            )(self._invoke_internal)
        else:
            self._invoke_with_retry = self._invoke_internal

    def _invoke_internal(self, messages: List[BaseMessage]) -> Any:
        """Internal invoke method that may be wrapped with retry."""
        return self.model.invoke(messages)

    def invoke(self, messages: List[BaseMessage]) -> Any:
        """
        Invoke the model with automatic retry.

        Args:
            messages: List of messages to send to the model

        Returns:
            Model response

        Raises:
            Exception: If all retry attempts fail
        """
        try:
            return self._invoke_with_retry(messages)
        except Exception as e:
            logger.error(f"Vision call failed after {self.max_retries} retries: {e}")
            raise

    @property
    def content(self):
        """Proxy content attribute for compatibility."""
        return getattr(self.model, 'content', None)


def create_vision_model_with_retry(
    llm_config: Dict[str, Any],
    max_retries: int = 3,
    **overrides
) -> VisionModelWithRetry:
    """
    Create a vision model with automatic retry logic.

    Args:
        llm_config: LLM configuration dict
        max_retries: Maximum retry attempts (default: 3)
        **overrides: Additional parameters to override

    Returns:
        VisionModelWithRetry wrapper around ChatOpenAI
    """
    base_model = LLMFactory.create_vision_model(llm_config, **overrides)
    return VisionModelWithRetry(base_model, max_retries=max_retries)
