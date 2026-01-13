"""
RLM - Recursive Language Models

A Python implementation of the Recursive Language Models approach
for handling arbitrarily long contexts with LLMs.
"""

from .rlm_core import RLM
from .client import OllamaClient
from .repl import REPLEnvironment

# Tool-based variant (for servers with tool calling)
from .rlm_tools import RLMTools
from .client_tools import OllamaChatClient
from .tools import ContextTools

__version__ = "0.1.0"
__all__ = [
    "RLM", "OllamaClient", "REPLEnvironment",
    "RLMTools", "OllamaChatClient", "ContextTools"
]
