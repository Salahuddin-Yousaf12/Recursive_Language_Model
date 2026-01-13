# RLM - Recursive Language Models

Implementation of the Recursive Language Models approach from MIT CSAIL paper.

## Project Structure

```
RLM/
├── rlm/                    # Core package
│   ├── __init__.py         # Package exports
│   ├── client.py           # Ollama API client (generate endpoint)
│   ├── client_tools.py     # Ollama chat client with tool support
│   ├── repl.py             # Python REPL environment
│   ├── rlm_core.py         # REPL-based RLM orchestrator
│   ├── rlm_tools.py        # Tool-based RLM orchestrator
│   ├── tools.py            # Context access tools
│   └── prompts.py          # System prompts
├── examples/               # Example scripts
│   ├── simple_query.py     # Basic usage
│   └── document_search.py  # Multi-document search
├── main.py                 # CLI interface
├── config.py               # Configuration settings
└── requirements.txt        # Dependencies
```

## Configuration

Edit `config.py` to change:
- `OLLAMA_API_URL`: API endpoint (default: https://ollama-ijcare-gpt.sheikhibrar.com/api/generate)
- `OLLAMA_MODEL`: Model name (default: gpt-oss:20b)
- `MAX_ITERATIONS`: Max LLM loops (default: 20)
- `MAX_RECURSION_DEPTH`: Sub-LLM call depth (default: 3)
- `CALL_DELAY`: Seconds to wait after each LLM call (default: 15) - important for thinking models

## Two Modes

### Tool-Based Mode (Default)
Uses native tool calling via the chat API. Works with servers that have tool calling enabled.

**Available tools:**
- `get_context_info`: Get overview and preview
- `read_chunk(index)`: Read a specific chunk
- `search(query)`: Find text in context
- `final_answer(answer)`: Provide the final answer

### REPL-Based Mode (--repl flag)
Uses Python code execution. Works with servers that don't intercept tool calls.

## Usage

### CLI
```bash
# Tool-based mode (default)
python main.py "Your query" -f context.txt -v --stats

# REPL-based mode
python main.py "Your query" -f context.txt -v --repl

# With options
python main.py "Your query" -f doc.txt --chunk-size 3000 --call-delay 10
```

### Python
```python
# Tool-based (recommended)
from rlm import RLMTools

rlm = RLMTools(verbose=True)
answer = rlm.query("Find the answer", context_string)

# REPL-based
from rlm import RLM

rlm = RLM(verbose=True)
answer = rlm.query("Find the answer", context_string)
```

## How It Works

**Core Insight:** Context is stored locally. The LLM explores it programmatically via tools, requesting only relevant chunks. This bypasses context window limitations.

### Tool-Based Flow

```
┌─────────────────────────────────────────────────────┐
│                  YOUR MACHINE                        │
│  ┌───────────────────────────────────────────────┐  │
│  │     CONTEXT (stored locally, never sent       │  │
│  │              to LLM all at once)              │  │
│  └───────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼                            │
│  ┌───────────────────────────────────────────────┐  │
│  │     TOOL HANDLER executes tool calls:         │  │
│  │     - search("keyword") → matches + positions │  │
│  │     - read_chunk(2) → chunk content           │  │
│  │     - final_answer("result") → done           │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
            Only small results sent back
                         ▼
┌─────────────────────────────────────────────────────┐
│                  OLLAMA SERVER                       │
│  ┌───────────────────────────────────────────────┐  │
│  │  LLM decides which tools to call:             │  │
│  │  1. search("magic number") → found at pos 847 │  │
│  │  2. read_chunk(2) → sees the relevant text    │  │
│  │  3. final_answer("42") → done                 │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Why This Enables "Unlimited" Context

1. Full context (even 1MB+) stays on your machine
2. LLM only receives small chunks when it requests them
3. Each tool call returns focused, relevant data
4. LLM's context window holds: query + tool definitions + accumulated results
5. The actual document size doesn't matter - only chunk size does

### Conversation Flow

1. **Send:** Query + tool definitions (not full context)
2. **LLM responds:** Tool call request (e.g., `search(query="...")`)
3. **Execute locally:** Run tool, get result
4. **Send back:** Tool result as message
5. **Repeat:** Until LLM calls `final_answer()`

## The Tools

Functions the LLM can call to explore your context:

| Tool | What It Does |
|------|--------------|
| `get_context_info()` | Returns total length, chunk count, 500-char preview |
| `read_chunk(index)` | Returns full text of chunk #index |
| `search(query)` | Finds matches, returns excerpts with positions |
| `final_answer(answer)` | Signals completion with the answer |

## REPL vs Tools: Two Approaches, Same Concept

### Original REPL Approach (--repl flag)
LLM writes Python code to explore context:
```python
print(len(context))           # Check size
print(context[0:500])         # Read start
idx = context.find("magic")   # Search
print(context[idx:idx+100])   # Read around match
```
We execute it, send output back. Repeat until `FINAL(answer)`.

### Tool Approach (default)
LLM calls predefined functions:
```
get_context_info()        → returns size, preview
search("magic")           → returns matches with positions
read_chunk(2)             → returns chunk content
final_answer("42")        → done
```
We execute handlers locally, send results back. Repeat until `final_answer()`.

**Same concept, different mechanism.** Both let the LLM programmatically explore data it can't see all at once.

## Why Tools/REPL Are Vital

**The Problem:**
```
LLM Context Window: ~32KB
Your Document: 2MB

You can't fit the document in the prompt.
```

**Traditional Solutions (Lossy):**
- Truncate → Lose information
- Summarize → Lose details
- RAG embeddings → Might miss relevant chunks

**RLM Solution:**
Don't send the document. Send tools to explore it.

```
What goes to LLM:          What stays local:
├── Query (50 tokens)      └── Full 2MB document
├── Tool definitions
├── Tool results (~500 tokens each)
└── Total: <2000 tokens ✓     (never sent whole)
```

The LLM becomes a **navigator**, not a reader. It decides what to look at; actual data stays local until requested.

## Code Flow

```
rlm_tools.py                    client_tools.py              Ollama Server
     │                                │                           │
     │  tools = get_tool_definitions()│                           │
     │  (from tools.py)               │                           │
     │                                │                           │
     ├──► client.chat(messages, ─────►├──► POST /api/chat ───────►│
     │        tools)                  │                           │
     │                                │◄── {tool_calls: [...]} ◄──┤
     │◄── response ◄──────────────────┤                           │
     │                                │                           │
     │  ctx_tools.handle_tool_call()  │                           │
     │  (executes locally in tools.py)│                           │
     │                                │                           │
     ├──► client.chat(messages + ────►├──► POST /api/chat ───────►│
     │        tool_result)            │                           │
     │                                │◄── {tool_calls or answer}◄┤
     │◄─────────────────────────────────                          │
     │                                │                           │
    ... repeats until final_answer() ...
```

**Where things are defined:**
- `tools.py:53-95` → Tool JSON schemas (sent to LLM)
- `tools.py:139-200` → Tool handlers (run locally)
- `client_tools.py:41-58` → Sends tools to Ollama
- `rlm_tools.py:119-170` → Main loop orchestrating everything

## Key Files

- **client_tools.py**: Chat API with tool calling support
- **tools.py**: Tool definitions + handlers (search, read_chunk, etc.)
- **rlm_tools.py**: Tool-based orchestrator (default mode)
- **rlm_core.py**: REPL-based orchestrator (--repl mode)
- **repl.py**: Python REPL environment for code execution
