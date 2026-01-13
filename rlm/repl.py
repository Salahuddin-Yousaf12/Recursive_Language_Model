"""
REPL Environment for RLM

Provides a sandboxed Python execution environment with:
- Persistent namespace across executions
- Pre-loaded context variable
- llm_query() function for recursive LLM calls
- Output capture and truncation
"""

import sys
import io
import traceback
import signal
import threading
from typing import Any, Callable, Optional
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TimeoutError(Exception):
    """Raised when code execution times out"""
    pass


class REPLEnvironment:
    """
    A sandboxed Python REPL environment for RLM.

    Maintains a persistent namespace and provides:
    - context: The user's input context
    - llm_query(): Function for recursive LLM calls
    - Standard Python builtins and common imports
    """

    def __init__(
        self,
        context: Any,
        llm_query_fn: Callable[[str], str],
        verbose: bool = False,
        max_output_length: int = None,
        execution_timeout: int = None
    ):
        """
        Initialize the REPL environment.

        Args:
            context: The context to make available (string, list, dict, etc.)
            llm_query_fn: Function to call for recursive LLM queries
            verbose: Whether to print debug information
            max_output_length: Max chars to return from execution
            execution_timeout: Max seconds for code execution
        """
        self.verbose = verbose
        self.max_output_length = max_output_length or config.MAX_OUTPUT_LENGTH
        self.execution_timeout = execution_timeout or config.CODE_EXECUTION_TIMEOUT

        # Track sub-LLM calls
        self.llm_call_count = 0

        # Create wrapper for llm_query that tracks calls
        def tracked_llm_query(prompt: str) -> str:
            self.llm_call_count += 1
            if self.verbose:
                print(f"[REPL] Sub-LLM call #{self.llm_call_count}")
            return llm_query_fn(prompt)

        # Initialize the namespace with context and llm_query
        self.namespace = {
            # The context variable
            "context": context,

            # The recursive LLM query function
            "llm_query": tracked_llm_query,

            # Common imports pre-loaded
            "re": __import__("re"),
            "json": __import__("json"),
            "math": __import__("math"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),

            # Built-in functions
            "__builtins__": __builtins__,
        }

        # Try to add optional useful modules
        try:
            self.namespace["requests"] = __import__("requests")
        except ImportError:
            pass

        if self.verbose:
            context_preview = str(context)[:200] + "..." if len(str(context)) > 200 else str(context)
            print(f"[REPL] Initialized with context: {context_preview}")

    def execute(self, code: str) -> tuple[str, bool]:
        """
        Execute Python code in the REPL environment.

        Args:
            code: The Python code to execute

        Returns:
            Tuple of (output_string, success_bool)
        """
        if self.verbose:
            print(f"[REPL] Executing code:\n{code[:500]}{'...' if len(code) > 500 else ''}")

        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output

        success = True
        error_message = None

        try:
            # Execute with timeout using threading
            result = {"done": False, "error": None}

            def run_code():
                try:
                    exec(code, self.namespace)
                    result["done"] = True
                except Exception as e:
                    result["error"] = e

            thread = threading.Thread(target=run_code)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.execution_timeout)

            if not result["done"] and result["error"] is None:
                success = False
                error_message = f"Code execution timed out after {self.execution_timeout} seconds"
                captured_output.write(f"\nTimeoutError: {error_message}")
            elif result["error"]:
                raise result["error"]

        except SyntaxError as e:
            success = False
            error_message = f"SyntaxError: {e}"
            captured_output.write(f"\n{error_message}")
            captured_output.write(f"\n{traceback.format_exc()}")

        except Exception as e:
            success = False
            error_message = f"{type(e).__name__}: {e}"
            captured_output.write(f"\n{error_message}")
            captured_output.write(f"\n{traceback.format_exc()}")

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Get output and truncate if necessary
        output = captured_output.getvalue()

        if len(output) > self.max_output_length:
            truncated_msg = f"\n\n[Output truncated. Showing first {self.max_output_length} chars of {len(output)} total]"
            output = output[:self.max_output_length] + truncated_msg

        if self.verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"[REPL] Execution {status}")
            if error_message:
                print(f"[REPL] Error: {error_message}")

        return output, success

    def get_variable(self, name: str) -> Any:
        """
        Get a variable from the namespace.

        Args:
            name: Variable name

        Returns:
            The variable value, or None if not found
        """
        return self.namespace.get(name)

    def set_variable(self, name: str, value: Any):
        """
        Set a variable in the namespace.

        Args:
            name: Variable name
            value: Variable value
        """
        self.namespace[name] = value

    def get_context_info(self) -> dict:
        """
        Get information about the context.

        Returns:
            Dictionary with context metadata
        """
        context = self.namespace.get("context", "")

        if isinstance(context, str):
            return {
                "type": "string",
                "total_length": len(context),
                "chunks": None
            }
        elif isinstance(context, list):
            return {
                "type": "list",
                "total_length": sum(len(str(item)) for item in context),
                "chunks": [len(str(item)) for item in context]
            }
        elif isinstance(context, dict):
            return {
                "type": "dict",
                "total_length": sum(len(str(v)) for v in context.values()),
                "chunks": {k: len(str(v)) for k, v in context.items()}
            }
        else:
            return {
                "type": type(context).__name__,
                "total_length": len(str(context)),
                "chunks": None
            }

    def get_stats(self) -> dict:
        """Get REPL statistics"""
        return {
            "llm_call_count": self.llm_call_count,
            "namespace_variables": list(self.namespace.keys())
        }


# Simple test
if __name__ == "__main__":
    # Mock LLM query function
    def mock_llm_query(prompt: str) -> str:
        return f"[Mock LLM Response to: {prompt[:50]}...]"

    # Test the REPL
    repl = REPLEnvironment(
        context="This is a test context with some information. The magic number is 42.",
        llm_query_fn=mock_llm_query,
        verbose=True
    )

    # Test basic execution
    output, success = repl.execute('''
print("Context length:", len(context))
print("First 50 chars:", context[:50])
''')
    print(f"\nOutput:\n{output}")
    print(f"Success: {success}")

    # Test llm_query
    output, success = repl.execute('''
answer = llm_query("What is the magic number?")
print("LLM said:", answer)
''')
    print(f"\nOutput:\n{output}")
    print(f"Success: {success}")

    # Test error handling
    output, success = repl.execute('''
undefined_variable
''')
    print(f"\nOutput:\n{output}")
    print(f"Success: {success}")

    print(f"\nStats: {repl.get_stats()}")
