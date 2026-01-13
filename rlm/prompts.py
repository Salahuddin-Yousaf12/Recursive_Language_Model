"""
System Prompts for RLM

Adapted from the RLM paper (Appendix D) for use with Ollama models.
"""


def get_rlm_system_prompt(
    context_type: str,
    context_total_length: int,
    context_lengths: list = None
) -> str:
    """
    Generate the RLM system prompt with context metadata.

    Args:
        context_type: Type of context (e.g., "string", "list of documents")
        context_total_length: Total character length of context
        context_lengths: Optional list of chunk lengths if context is chunked
    """

    lengths_str = str(context_lengths) if context_lengths else "single chunk"

    return f'''You are an AI that answers queries using a Python REPL environment.

CONTEXT INFO: {context_type}, {context_total_length:,} characters ({lengths_str})

AVAILABLE IN REPL:
- context: variable containing the data to analyze
- llm_query(prompt): function to ask questions about text
- print(): to see output

HOW TO RESPOND:
1. Write Python code in ```python blocks to explore the context
2. Use print() to see results
3. When done, write FINAL(your answer here)

EXAMPLE:
```python
# Look at the context
print(context[:500])

# For complex analysis, use llm_query
answer = llm_query(f"What is important in this text: {{context}}")
print(answer)
```

Then provide: FINAL(the answer based on what you found)

Remember: Write actual code to explore, then FINAL(answer) when done.'''


def get_continuation_prompt(
    previous_output: str,
    iteration: int
) -> str:
    """
    Generate continuation prompt after code execution.

    Args:
        previous_output: Output from the previous code execution
        iteration: Current iteration number
    """
    return f'''[Iteration {iteration}] The REPL executed your code. Here is the output:

{previous_output}

Continue your analysis. You can:
1. Write more code in ```repl``` blocks to further explore the context
2. Make more llm_query() calls to analyze specific parts
3. Provide your final answer with FINAL(answer) or FINAL_VAR(variable_name)

What would you like to do next?'''


def get_error_prompt(error_message: str) -> str:
    """
    Generate prompt when code execution fails.
    """
    return f'''Your code encountered an error:

{error_message}

Please fix the code and try again. Remember to:
- Check for syntax errors
- Ensure variables are defined before use
- Handle potential edge cases

Write corrected code in a ```repl``` block.'''
