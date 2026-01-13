"""
Ollama Chat API Client with Tool Support for RLM
"""

import requests
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class OllamaChatClient:
    """Client for Ollama Chat API with tool calling support"""

    def __init__(
        self,
        api_url: str = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        call_delay: float = None,
        verbose: bool = False
    ):
        # Use chat endpoint instead of generate
        base_url = api_url or config.OLLAMA_API_URL
        self.api_url = base_url.replace("/api/generate", "/api/chat")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.max_retries = max_retries or config.MAX_RETRIES
        self.call_delay = call_delay if call_delay is not None else config.CALL_DELAY
        self.verbose = verbose

        # Track usage
        self.total_calls = 0
        self.total_tool_calls = 0

    def chat(
        self,
        messages: list,
        tools: list = None,
        temperature: float = 0.7
    ) -> dict:
        """
        Send a chat request with optional tools.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            temperature: Sampling temperature

        Returns:
            The response message dict (may contain tool_calls)
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }

        if tools:
            payload["tools"] = tools

        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"[ChatClient] Request attempt {attempt + 1}/{self.max_retries}")
                    print(f"[ChatClient] Messages: {len(messages)}, Tools: {len(tools) if tools else 0}")

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                message = result.get("message", {})

                self.total_calls += 1
                if message.get("tool_calls"):
                    self.total_tool_calls += len(message["tool_calls"])

                if self.verbose:
                    if message.get("tool_calls"):
                        print(f"[ChatClient] Got {len(message['tool_calls'])} tool call(s)")
                    elif message.get("content"):
                        print(f"[ChatClient] Got response: {len(message['content'])} chars")

                # Wait after successful call
                if self.call_delay > 0:
                    if self.verbose:
                        print(f"[ChatClient] Waiting {self.call_delay}s...")
                    time.sleep(self.call_delay)

                return message

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                if self.verbose:
                    print(f"[ChatClient] {last_error}")

            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
                if self.verbose:
                    print(f"[ChatClient] {last_error}")

            if attempt < self.max_retries - 1:
                time.sleep(config.RETRY_DELAY)

        raise RuntimeError(f"Failed after {self.max_retries} attempts: {last_error}")

    def get_stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tool_calls": self.total_tool_calls
        }


if __name__ == "__main__":
    client = OllamaChatClient(verbose=True)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }]

    messages = [{"role": "user", "content": "What's the weather in Paris?"}]
    response = client.chat(messages, tools)
    print(f"\nResponse: {response}")
