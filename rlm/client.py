"""
Ollama API Client for RLM
"""

import requests
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(
        self,
        api_url: str = None,
        model: str = None,
        timeout: int = None,
        max_retries: int = None,
        call_delay: float = None,
        verbose: bool = False
    ):
        self.api_url = api_url or config.OLLAMA_API_URL
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.max_retries = max_retries or config.MAX_RETRIES
        self.call_delay = call_delay if call_delay is not None else config.CALL_DELAY
        self.verbose = verbose

        # Track usage statistics
        self.total_calls = 0
        self.total_prompt_chars = 0
        self.total_response_chars = 0

    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = None
    ) -> str:
        """
        Generate a response from the Ollama API.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate (None for default)

        Returns:
            The generated response text
        """
        # Combine system and user prompt into a single prompt
        # This works better with some Ollama deployments
        if system:
            full_prompt = f"### System:\n{system}\n\n### User:\n{prompt}\n\n### Assistant:"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        # Note: We don't use the 'system' field anymore - it's embedded in prompt

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if self.verbose:
                    print(f"[OllamaClient] Sending request (attempt {attempt + 1}/{self.max_retries})...")
                    print(f"[OllamaClient] Prompt length: {len(full_prompt)} chars")

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")

                # Update statistics
                self.total_calls += 1
                self.total_prompt_chars += len(full_prompt)
                self.total_response_chars += len(response_text)

                if self.verbose:
                    print(f"[OllamaClient] Response received: {len(response_text)} chars")

                # Wait after successful call (for thinking models that need cooldown)
                if self.call_delay > 0:
                    if self.verbose:
                        print(f"[OllamaClient] Waiting {self.call_delay}s after call (thinking model cooldown)...")
                    time.sleep(self.call_delay)

                return response_text

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout} seconds"
                if self.verbose:
                    print(f"[OllamaClient] {last_error}")

            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                if self.verbose:
                    print(f"[OllamaClient] {last_error}")

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                time.sleep(config.RETRY_DELAY)

        raise RuntimeError(f"Failed after {self.max_retries} attempts. Last error: {last_error}")

    def get_stats(self) -> dict:
        """Return usage statistics"""
        return {
            "total_calls": self.total_calls,
            "total_prompt_chars": self.total_prompt_chars,
            "total_response_chars": self.total_response_chars
        }

    def reset_stats(self):
        """Reset usage statistics"""
        self.total_calls = 0
        self.total_prompt_chars = 0
        self.total_response_chars = 0


# Simple test
if __name__ == "__main__":
    client = OllamaClient(verbose=True)
    response = client.generate("Say hello in one sentence.")
    print(f"\nResponse: {response}")
    print(f"\nStats: {client.get_stats()}")
