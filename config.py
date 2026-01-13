"""
RLM Configuration Settings
"""

# Ollama API Configuration
OLLAMA_API_URL = "https://ollama-ijcare-gpt.sheikhibrar.com/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"

# Request Settings
REQUEST_TIMEOUT = 300  # seconds (increased for thinking models)
MAX_RETRIES = 5  # More retries for unreliable APIs
RETRY_DELAY = 15  # seconds between retries
CALL_DELAY = 15  # seconds to wait between successful LLM calls (for thinking models)

# RLM Settings
MAX_ITERATIONS = 20  # Maximum number of LLM interaction loops
MAX_RECURSION_DEPTH = 3  # Maximum depth for sub-LLM calls
MAX_OUTPUT_LENGTH = 10000  # Maximum characters to show from code execution
CODE_EXECUTION_TIMEOUT = 30  # seconds per code block

# Context Settings
CONTEXT_PREVIEW_LENGTH = 500  # Characters to show in context preview
MAX_CONTEXT_IN_PROMPT = 50000  # Max context chars to include directly in prompt

# Verbosity
DEFAULT_VERBOSE = False
