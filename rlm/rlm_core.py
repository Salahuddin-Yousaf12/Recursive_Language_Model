"""
RLM Core - Recursive Language Model Orchestrator

The main class that ties together:
- Ollama client for LLM calls
- REPL environment for code execution
- Iterative loop for processing queries
"""

import re
import sys
import os
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rlm.client import OllamaClient
from rlm.repl import REPLEnvironment
from rlm import prompts


class RLM:
    """
    Recursive Language Model

    Processes arbitrarily long contexts by treating them as external
    environment variables that the LLM can programmatically interact with.
    """

    # Regex patterns for parsing LLM responses
    CODE_PATTERN = re.compile(r"```(?:repl|python)\n(.*?)```", re.DOTALL)
    FINAL_PATTERN = re.compile(r"FINAL\((.*?)\)", re.DOTALL)
    FINAL_VAR_PATTERN = re.compile(r"FINAL_VAR\((\w+)\)")

    def __init__(
        self,
        api_url: str = None,
        model: str = None,
        verbose: bool = None,
        max_iterations: int = None,
        max_recursion_depth: int = None,
        call_delay: float = None
    ):
        """
        Initialize the RLM.

        Args:
            api_url: Ollama API URL (defaults to config)
            model: Model name (defaults to config)
            verbose: Whether to print detailed output
            max_iterations: Maximum LLM interaction loops
            max_recursion_depth: Maximum depth for sub-LLM calls
            call_delay: Seconds to wait between LLM calls (for thinking models)
        """
        self.verbose = verbose if verbose is not None else config.DEFAULT_VERBOSE
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.max_recursion_depth = max_recursion_depth or config.MAX_RECURSION_DEPTH

        # Initialize the Ollama client
        self.client = OllamaClient(
            api_url=api_url,
            model=model,
            call_delay=call_delay,
            verbose=self.verbose
        )

        # Track statistics
        self.last_iteration_count = 0
        self.last_sub_llm_calls = 0

    def _log(self, message: str, prefix: str = "RLM"):
        """Print a log message if verbose mode is enabled"""
        if self.verbose:
            print(f"[{prefix}] {message}")

    def _create_llm_query_fn(self, depth: int) -> callable:
        """
        Create a llm_query function for a given recursion depth.

        Args:
            depth: Current recursion depth

        Returns:
            A function that can be used for sub-LLM queries
        """
        def llm_query(prompt: str) -> str:
            if depth >= self.max_recursion_depth:
                return f"[ERROR: Maximum recursion depth ({self.max_recursion_depth}) reached. Cannot make more sub-LLM calls.]"

            self._log(f"Sub-LLM call at depth {depth + 1}", "SUB-LLM")

            # For sub-calls, we use a simpler prompt without the REPL instructions
            response = self.client.generate(
                prompt=prompt,
                system="You are a helpful assistant. Answer the question concisely and accurately."
            )

            return response

        return llm_query

    def _extract_code_blocks(self, response: str) -> list[str]:
        """Extract code blocks from LLM response"""
        matches = self.CODE_PATTERN.findall(response)
        return matches

    def _check_final_answer(self, response: str, repl: REPLEnvironment) -> tuple[bool, Optional[str]]:
        """
        Check if the response contains a final answer.

        Returns:
            Tuple of (is_final, answer)
        """
        # Check for FINAL_VAR first (more specific)
        var_match = self.FINAL_VAR_PATTERN.search(response)
        if var_match:
            var_name = var_match.group(1)
            var_value = repl.get_variable(var_name)
            if var_value is not None:
                self._log(f"Final answer from variable '{var_name}'")
                return True, str(var_value)
            else:
                self._log(f"Warning: FINAL_VAR referenced undefined variable '{var_name}'")

        # Check for FINAL()
        final_match = self.FINAL_PATTERN.search(response)
        if final_match:
            answer = final_match.group(1).strip()
            self._log(f"Final answer provided directly")
            return True, answer

        return False, None

    def query(self, query: str, context: Any) -> str:
        """
        Process a query with the given context using the RLM approach.

        Args:
            query: The user's question/query
            context: The context (string, list, dict, etc.)

        Returns:
            The final answer string
        """
        self._log(f"Starting RLM query")
        self._log(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")

        # Create the REPL environment
        llm_query_fn = self._create_llm_query_fn(depth=0)
        repl = REPLEnvironment(
            context=context,
            llm_query_fn=llm_query_fn,
            verbose=self.verbose
        )

        # Get context metadata
        context_info = repl.get_context_info()
        self._log(f"Context: {context_info['type']}, {context_info['total_length']:,} chars")

        # Build the initial prompt
        system_prompt = prompts.get_rlm_system_prompt(
            context_type=context_info["type"],
            context_total_length=context_info["total_length"],
            context_lengths=context_info.get("chunks")
        )

        # Construct the user message
        # For small contexts, include a preview in the prompt
        context_str = str(context)
        if len(context_str) < 5000:
            context_preview = context_str
        else:
            context_preview = context_str[:2000] + f"\n\n[... {len(context_str) - 2000} more characters ...]"

        user_message = f"""QUERY: {query}

CONTEXT PREVIEW (use 'context' variable in code for full access):
{context_preview}

Write Python code in ```python blocks to analyze the context, then FINAL(your answer)."""

        # Main iteration loop
        iteration = 0
        conversation_context = ""  # Track execution history for the LLM

        while iteration < self.max_iterations:
            iteration += 1
            self._log(f"=== Iteration {iteration}/{self.max_iterations} ===")

            # Build the full prompt
            if iteration == 1:
                full_prompt = user_message
                if conversation_context:
                    full_prompt += f"\n\n{conversation_context}"
            else:
                full_prompt = conversation_context

            # Call the LLM
            response = self.client.generate(
                prompt=full_prompt,
                system=system_prompt
            )

            self._log(f"LLM response length: {len(response)} chars")

            # Check for final answer
            is_final, answer = self._check_final_answer(response, repl)
            if is_final:
                self.last_iteration_count = iteration
                self.last_sub_llm_calls = repl.llm_call_count
                self._log(f"Completed in {iteration} iterations with {repl.llm_call_count} sub-LLM calls")
                return answer

            # Extract and execute code blocks
            code_blocks = self._extract_code_blocks(response)

            if code_blocks:
                self._log(f"Found {len(code_blocks)} code block(s)")

                all_outputs = []
                for i, code in enumerate(code_blocks):
                    self._log(f"Executing code block {i + 1}")
                    output, success = repl.execute(code)

                    status = "SUCCESS" if success else "ERROR"
                    all_outputs.append(f"[Code Block {i + 1} - {status}]\n{output}")

                # Update conversation context with execution results
                execution_output = "\n\n".join(all_outputs)
                conversation_context = prompts.get_continuation_prompt(
                    previous_output=execution_output,
                    iteration=iteration
                )

            else:
                # No code blocks found - the LLM might be thinking or confused
                self._log("No code blocks found in response")

                # Check if the response seems like a final answer even without FINAL()
                if len(response) < 500 and not any(keyword in response.lower() for keyword in ["```", "code", "execute", "let me"]):
                    self._log("Response appears to be a direct answer (no FINAL tag)")
                    self.last_iteration_count = iteration
                    self.last_sub_llm_calls = repl.llm_call_count
                    return response.strip()

                # Prompt the LLM to continue
                conversation_context = f"""Your previous response did not contain any code blocks or a FINAL() answer.

Previous response:
{response[:2000]}{'...' if len(response) > 2000 else ''}

Please either:
1. Write code in ```repl``` blocks to explore the context
2. Provide your final answer with FINAL(your answer here)

What would you like to do?"""

        # Max iterations reached
        self._log(f"Warning: Max iterations ({self.max_iterations}) reached without final answer")
        self.last_iteration_count = iteration
        self.last_sub_llm_calls = repl.llm_call_count

        # Try to extract any useful information from the last response
        return f"[Max iterations reached. Last LLM response:]\n{response[:1000]}"

    def get_stats(self) -> dict:
        """Get statistics from the last query"""
        return {
            "last_iteration_count": self.last_iteration_count,
            "last_sub_llm_calls": self.last_sub_llm_calls,
            "client_stats": self.client.get_stats()
        }


# Simple test
if __name__ == "__main__":
    print("Testing RLM with a simple query...")
    print("=" * 50)

    rlm = RLM(verbose=True)

    # Test with a simple context
    context = """
    Document 1: The company was founded in 2010.
    Document 2: The CEO is John Smith.
    Document 3: The headquarters is in New York.
    Document 4: The magic number for accessing the vault is 42.
    Document 5: Annual revenue is $10 million.
    """

    query = "What is the magic number for accessing the vault?"

    try:
        answer = rlm.query(query, context)
        print("\n" + "=" * 50)
        print(f"FINAL ANSWER: {answer}")
        print(f"\nStats: {rlm.get_stats()}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
