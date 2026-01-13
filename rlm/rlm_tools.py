"""
Tool-Based RLM - Recursive Language Model using Native Tool Calling

Instead of asking the LLM to write Python code (which conflicts with
server-side tool parsing), this version uses native tool calls to
let the LLM programmatically explore arbitrarily large contexts.

The key insight remains: context is stored locally, and only relevant
chunks are sent to the LLM when it requests them via tools.
"""

import sys
import os
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rlm.client_tools import OllamaChatClient
from rlm.tools import ContextTools


class RLMTools:
    """
    Tool-Based Recursive Language Model

    Uses native tool calling to let the LLM explore large contexts
    without hitting context window limits.
    """

    def __init__(
        self,
        api_url: str = None,
        model: str = None,
        verbose: bool = None,
        max_iterations: int = None,
        max_recursion_depth: int = None,
        call_delay: float = None,
        chunk_size: int = 2000
    ):
        self.verbose = verbose if verbose is not None else config.DEFAULT_VERBOSE
        self.max_iterations = max_iterations or config.MAX_ITERATIONS
        self.max_recursion_depth = max_recursion_depth or config.MAX_RECURSION_DEPTH
        self.chunk_size = chunk_size

        self.client = OllamaChatClient(
            api_url=api_url,
            model=model,
            call_delay=call_delay,
            verbose=self.verbose
        )

        # Stats
        self.last_iteration_count = 0
        self.last_tool_calls = 0

    def _log(self, msg: str, prefix: str = "RLM"):
        if self.verbose:
            print(f"[{prefix}] {msg}")

    def _create_llm_query_fn(self, depth: int):
        """Create function for sub-LLM queries (used by ask_about_chunk tool)"""
        def llm_query(prompt: str) -> str:
            if depth >= self.max_recursion_depth:
                return f"[Max recursion depth ({self.max_recursion_depth}) reached]"

            self._log(f"Sub-LLM call at depth {depth + 1}", "SUB-LLM")

            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat(messages, tools=None)
            return response.get("content", "")

        return llm_query

    def query(self, query: str, context: Any) -> str:
        """
        Process a query against the given context using tool-based exploration.

        Args:
            query: The user's question
            context: The context to search (string, list, or dict)

        Returns:
            The final answer string
        """
        self._log(f"Starting tool-based RLM query")
        self._log(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")

        # Initialize context tools
        llm_query_fn = self._create_llm_query_fn(depth=0)
        ctx_tools = ContextTools(
            context=context,
            llm_query_fn=llm_query_fn,
            chunk_size=self.chunk_size,
            verbose=self.verbose
        )

        tool_definitions = ContextTools.get_tool_definitions()
        stats = ctx_tools.get_stats()
        self._log(f"Context: {stats['context_length']:,} chars, {stats['num_chunks']} chunks")

        # Build initial system message
        system_content = f"""You answer questions by exploring a context using tools. You MUST use the final_answer tool to provide your answer.

CONTEXT: {stats['context_length']:,} characters in {stats['num_chunks']} chunks (indices 0-{stats['num_chunks']-1})

TOOLS:
- get_context_info: Overview and preview
- read_chunk(chunk_index): Read chunk 0-{stats['num_chunks']-1}
- search(query): Find text, get excerpts with positions
- final_answer(answer): REQUIRED - call this with your answer

IMPORTANT: When you have enough information, you MUST call final_answer(answer). Do not just write text - use the tool."""

        # Initialize conversation
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        # Main loop
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self._log(f"=== Iteration {iteration}/{self.max_iterations} ===")

            # Get LLM response
            response = self.client.chat(messages, tools=tool_definitions)

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if tool_calls:
                self._log(f"Processing {len(tool_calls)} tool call(s)")

                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": tool_calls
                })

                # Process each tool call
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", {})

                    # Handle arguments that might be strings (JSON)
                    if isinstance(args, str):
                        import json
                        try:
                            args = json.loads(args)
                        except:
                            args = {}

                    result, is_final = ctx_tools.handle_tool_call(name, args)

                    if is_final:
                        self._log(f"Final answer received after {iteration} iterations")
                        self.last_iteration_count = iteration
                        self.last_tool_calls = ctx_tools.tool_call_count
                        return result

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "content": result
                    })

            else:
                # No tool calls - check if there's a text response
                content = response.get("content", "").strip()

                if content:
                    self._log(f"Got text response ({len(content)} chars) without tool call")

                    # If we've done some exploration and got a substantive response, accept it
                    if ctx_tools.tool_call_count > 0 and len(content) > 200:
                        self._log("Treating as implicit final answer (after exploration)")
                        self.last_iteration_count = iteration
                        self.last_tool_calls = ctx_tools.tool_call_count
                        return content

                    # Otherwise, prompt to call final_answer
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": "Now call final_answer(answer) with your complete answer."
                    })
                else:
                    # Empty response - prompt for action
                    messages.append({
                        "role": "user",
                        "content": "Use search(query) to find relevant information, then call final_answer(answer)."
                    })

        # Max iterations reached
        self._log(f"Max iterations ({self.max_iterations}) reached")
        self.last_iteration_count = iteration
        self.last_tool_calls = ctx_tools.tool_call_count

        return f"[Max iterations reached. Could not find definitive answer.]"

    def get_stats(self) -> dict:
        return {
            "last_iteration_count": self.last_iteration_count,
            "last_tool_calls": self.last_tool_calls,
            "client_stats": self.client.get_stats()
        }


# Test
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("Testing Tool-Based RLM...")
    print("=" * 60)

    rlm = RLMTools(verbose=True)

    context = """
    Document 1: The company TechCorp was founded in 2010 by Dr. Alice Johnson.
    Document 2: The current CEO is Bob Smith, who was appointed in 2015.
    Document 3: Headquarters are located at 123 Innovation Drive, New York City.
    Document 4: The secret access code for the executive vault is DIAMOND-9876.
    Document 5: Annual revenue for 2023 was $50 million, up 20% from 2022.
    Document 6: The company employs 500 people across 12 countries.
    Document 7: Main products include CloudSync, DataVault, and SecureLink.
    Document 8: The R&D budget for 2024 is $8 million.
    Document 9: The company went public on NASDAQ in 2018 under ticker TECH.
    Document 10: Board of directors includes 7 members, chaired by Dr. Carol Williams.
    """

    query = "What is the secret access code for the executive vault?"

    try:
        answer = rlm.query(query, context)
        print("\n" + "=" * 60)
        print(f"FINAL ANSWER: {answer}")
        print(f"\nStats: {rlm.get_stats()}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
