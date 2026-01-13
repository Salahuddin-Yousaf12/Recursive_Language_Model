#!/usr/bin/env python3
"""
RLM CLI - Command Line Interface for Recursive Language Models

Usage:
    python main.py "Your query here" --context-file document.txt
    python main.py "Your query here" --context "Inline context here"
    python main.py "Your query here" -f document.txt -v
"""

import argparse
import sys
import os

# Handle Unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlm import RLM, RLMTools
import config


def main():
    parser = argparse.ArgumentParser(
        description="RLM - Recursive Language Models CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Find the magic number" -f document.txt
  python main.py "Summarize the key points" --context-file notes.txt -v
  python main.py "What is 2+2?" --context "Math context" --max-iterations 5
        """
    )

    # Required: Query
    parser.add_argument(
        "query",
        type=str,
        help="The question or query to answer"
    )

    # Context input (one of these is required)
    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument(
        "-f", "--context-file",
        type=str,
        help="Path to a file containing the context"
    )
    context_group.add_argument(
        "-c", "--context",
        type=str,
        help="Inline context string"
    )

    # Optional settings
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output showing execution details"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=config.MAX_ITERATIONS,
        help=f"Maximum number of LLM interaction loops (default: {config.MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=config.MAX_RECURSION_DEPTH,
        help=f"Maximum recursion depth for sub-LLM calls (default: {config.MAX_RECURSION_DEPTH})"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=config.OLLAMA_API_URL,
        help=f"Ollama API URL (default: {config.OLLAMA_API_URL})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.OLLAMA_MODEL,
        help=f"Model name (default: {config.OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--call-delay",
        type=float,
        default=config.CALL_DELAY,
        help=f"Seconds to wait between LLM calls - useful for thinking models (default: {config.CALL_DELAY})"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics after completion"
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Use REPL-based mode instead of tool-based (for servers without tool calling)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size for context splitting in tool mode (default: 2000)"
    )

    args = parser.parse_args()

    # Load context
    if args.context_file:
        try:
            with open(args.context_file, "r", encoding="utf-8") as f:
                context = f.read()
            if args.verbose:
                print(f"[CLI] Loaded context from {args.context_file} ({len(context):,} chars)")
        except FileNotFoundError:
            print(f"Error: File not found: {args.context_file}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        context = args.context

    # Initialize RLM
    mode = "REPL-based" if args.repl else "Tool-based"
    if args.verbose:
        print(f"[CLI] Initializing RLM ({mode} mode)...")
        print(f"[CLI] API URL: {args.api_url}")
        print(f"[CLI] Model: {args.model}")
        print(f"[CLI] Max iterations: {args.max_iterations}")
        print(f"[CLI] Max recursion depth: {args.max_depth}")
        print(f"[CLI] Call delay: {args.call_delay}s (wait after each LLM call)")
        if not args.repl:
            print(f"[CLI] Chunk size: {args.chunk_size} chars")
        print()

    if args.repl:
        # Original REPL-based mode
        rlm = RLM(
            api_url=args.api_url,
            model=args.model,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            max_recursion_depth=args.max_depth,
            call_delay=args.call_delay
        )
    else:
        # Tool-based mode (default - works with servers that have tool calling)
        rlm = RLMTools(
            api_url=args.api_url,
            model=args.model,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            max_recursion_depth=args.max_depth,
            call_delay=args.call_delay,
            chunk_size=args.chunk_size
        )

    # Run the query
    try:
        if args.verbose:
            print("=" * 60)
            print("STARTING RLM QUERY")
            print("=" * 60)
            print()

        answer = rlm.query(args.query, context)

        print()
        print("=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print()

        if args.stats:
            stats = rlm.get_stats()
            print("=" * 60)
            print("STATISTICS:")
            print("=" * 60)
            print(f"  Iterations: {stats['last_iteration_count']}")
            if args.repl:
                print(f"  Sub-LLM calls: {stats['last_sub_llm_calls']}")
                print(f"  Total LLM calls: {stats['client_stats']['total_calls']}")
                print(f"  Total prompt chars: {stats['client_stats']['total_prompt_chars']:,}")
                print(f"  Total response chars: {stats['client_stats']['total_response_chars']:,}")
            else:
                print(f"  Tool calls: {stats['last_tool_calls']}")
                print(f"  Total API calls: {stats['client_stats']['total_calls']}")
                print(f"  Total tool invocations: {stats['client_stats']['total_tool_calls']}")
            print()

    except KeyboardInterrupt:
        print("\n[CLI] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
