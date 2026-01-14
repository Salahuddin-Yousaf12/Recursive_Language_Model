# RLM - Recursive Language Models

Implementation of the Recursive Language Models approach from MIT CSAIL paper (arXiv:2512.24601).

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

---

## Understanding the Recursive Logic (Visual Guide)

### The Problem: LLMs Have Limited Context Windows

```
┌─────────────────────────────────────────────────────────────┐
│                YOUR DOCUMENT (e.g., 2MB file)               │
│                                                             │
│  ████████████████████████████████████████████████████████   │
│  ████████████████████████████████████████████████████████   │
│  ████████████████████████████████████████████████████████   │
│  ████████████████████████████████████████████████████████   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM CONTEXT WINDOW (limited!)                  │
│  ┌────────────────────┐                                     │
│  │  Can only fit      │   ← What if document is 10x bigger? │
│  │  THIS much         │      Or 100x? LLM chokes.           │
│  └────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

### Traditional Approach: Stuff It All In (FAILS)

```
    YOU: "Hey LLM, here's a 2MB document, find the magic number"

         ┌──────────────────────────────────┐
         │  2MB OF TEXT                     │
         │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
         │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │──────► LLM
         │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
         └──────────────────────────────────┘
                                                     │
                                                     ▼
                                              ❌ FAILS!
                                              - Doesn't fit
                                              - "Context rot"
                                              - Forgets stuff
```

### RLM Approach: LLM as EXPLORER, Not READER

**Key insight:** Don't send the document. Send TOOLS to explore it.

```
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR MACHINE (local)                        │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  context = "A decoder-only transformer begins..."       │    │
│   │           (FULL 2MB document stored as a variable)      │    │
│   │           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │    │
│   │           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │    │
│   └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              │  LLM can only see SMALL pieces    │
│                              ▼  at a time via code               │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  PYTHON REPL                                            │    │
│   │                                                         │    │
│   │  >>> print(len(context))                                │    │
│   │  76543                                                  │    │
│   │                                                         │    │
│   │  >>> print(context[0:500])     ◄── LLM "peeks" at start │    │
│   │  "A decoder-only transformer..."                        │    │
│   │                                                         │    │
│   │  >>> print(context.find("attention"))  ◄── LLM searches │    │
│   │  1842                                                   │    │
│   └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Only SMALL outputs go to LLM
                              ▼
                    ┌───────────────────┐
                    │       LLM         │
                    │  "Ah! Attention   │
                    │   is at position  │
                    │   1842. Let me    │
                    │   read around it" │
                    └───────────────────┘
```

### THE RECURSIVE PART: Sub-LLMs

The main LLM can **spawn helper LLMs** to analyze chunks:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ROOT LLM (the boss)                           │
│                                                                         │
│   "This document is 76,000 chars. Too big to understand at once.        │
│    I'll split it into chunks and ask SUB-LLMs to analyze each."         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Writes code:
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  for i in range(0, len(context), 10000):                                │
│      chunk = context[i:i+10000]                                         │
│      answer = llm_query(f"What's this chunk about? {chunk}")  ◄─────────│
│      results.append(answer)                                    │        │
└────────────────────────────────────────────────────────────────│────────┘
                                                                 │
                    ┌────────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                     SUB-LLM CALLS (depth=1)                   │
    │                                                               │
    │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
    │   │  Sub-LLM 1  │  │  Sub-LLM 2  │  │  Sub-LLM 3  │   ...    │
    │   │             │  │             │  │             │          │
    │   │ "Chunk 1 is │  │ "Chunk 2 is │  │ "Chunk 3 is │          │
    │   │  about      │  │  about      │  │  about      │          │
    │   │  tokeniza-  │  │  attention  │  │  training"  │          │
    │   │  tion"      │  │  mechanism" │  │             │          │
    │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
    │          │                │                │                 │
    └──────────│────────────────│────────────────│─────────────────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                                ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    ROOT LLM AGGREGATES                        │
    │                                                               │
    │   "Based on my sub-LLMs:                                      │
    │    - Chunk 1: tokenization                                    │
    │    - Chunk 2: attention                                       │
    │    - Chunk 3: training                                        │
    │                                                               │
    │    The document explains how transformers work!"              │
    │                                                               │
    │    FINAL("This is a technical explanation of transformers")   │
    └───────────────────────────────────────────────────────────────┘
```

### Step-by-Step Flow

```
STEP 1: You ask a question
─────────────────────────────────────────────────────────────
    YOU ──► "What is this document about?"
            + context.txt (stored locally, NOT sent to LLM)


STEP 2: RLM tells LLM about the environment
─────────────────────────────────────────────────────────────
    SYSTEM ──► "You have a variable 'context' with 76,543 chars.
               You can write Python code to explore it.
               You can call llm_query() to ask sub-LLMs."


STEP 3: LLM writes code to explore
─────────────────────────────────────────────────────────────
    LLM ──► ```python
            print(context[:1000])  # See the start
            print(context[-1000:]) # See the end
            ```


STEP 4: Code runs locally, output sent back
─────────────────────────────────────────────────────────────
    REPL ──► "A decoder-only transformer begins..."
             (only 1000 chars sent, not 76,543!)


STEP 5: LLM decides to use sub-LLMs for deep analysis
─────────────────────────────────────────────────────────────
    LLM ──► ```python
            summary = llm_query(f"Summarize: {context[0:20000]}")
            print(summary)
            ```


STEP 6: Sub-LLM processes the chunk
─────────────────────────────────────────────────────────────
    SUB-LLM ──► "This section explains tokenization and
                 embedding in transformers."


STEP 7: Root LLM gets result, continues or finishes
─────────────────────────────────────────────────────────────
    LLM ──► FINAL("The document is a technical deep-dive
                   explaining how decoder-only transformers work,
                   covering tokenization, attention, and training.")
```

### Why "Recursive"?

```
                    ┌─────────────────┐
                    │    ROOT LLM     │  ◄── Main "brain"
                    │   (depth = 0)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Sub-LLM  │   │ Sub-LLM  │   │ Sub-LLM  │  ◄── "Helper" brains
        │(depth=1) │   │(depth=1) │   │(depth=1) │
        └──────────┘   └──────────┘   └──────────┘

The LLM calls ITSELF (or a smaller version) on sub-problems.
That's the "recursive" part - like a function calling itself!
```

### What Gets Sent Where (The Magic)

```
┌─────────────────────────────────────────────────────────────────┐
│  STAYS ON YOUR MACHINE              │  GOES TO LLM (small!)    │
├─────────────────────────────────────┼──────────────────────────┤
│                                     │                          │
│  • Full 2MB document                │  • Your question         │
│  • All chunks                       │  • System prompt         │
│  • Search index                     │  • Tool definitions      │
│                                     │  • Small code outputs    │
│                                     │  • Sub-LLM responses     │
│                                     │                          │
│  ████████████████████████           │  ░░░░                    │
│  ████████████████████████           │                          │
│  ████████████████████████           │                          │
│                                     │                          │
└─────────────────────────────────────┴──────────────────────────┘
         HUGE (never sent)                    TINY (fits!)
```

---

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

**REPL Mode** (`--repl`): LLM writes actual Python code
```python
# LLM literally writes this:
chunk = context[5000:6000]
answer = llm_query(f"What topic? {chunk}")
print(answer)
```

**Tool Mode** (default): LLM calls predefined functions
```
LLM calls: search("attention")     → Returns matches
LLM calls: read_chunk(3)           → Returns chunk #3
LLM calls: final_answer("...")     → Done!
```

---

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

---

## How It Works (Technical)

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

---

## The Tools

Functions the LLM can call to explore your context:

| Tool | What It Does |
|------|--------------|
| `get_context_info()` | Returns total length, chunk count, 500-char preview |
| `read_chunk(index)` | Returns full text of chunk #index |
| `search(query)` | Finds matches, returns excerpts with positions |
| `final_answer(answer)` | Signals completion with the answer |

---

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

---

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

---

## Key Files

- **client_tools.py**: Chat API with tool calling support
- **tools.py**: Tool definitions + handlers (search, read_chunk, etc.)
- **rlm_tools.py**: Tool-based orchestrator (default mode)
- **rlm_core.py**: REPL-based orchestrator (--repl mode)
- **repl.py**: Python REPL environment for code execution

---

## Paper Reference

Based on: "Recursive Language Models" (arXiv:2512.24601, Dec 2025)
- Authors: Alex L. Zhang, Tim Kraska, Omar Khattab (MIT CSAIL)
- Key result: RLMs handle inputs up to 10M+ tokens (100x beyond context windows)
- Performance: 91.3% on BrowseComp+ vs 0% for base GPT-5 (can't fit in context)
