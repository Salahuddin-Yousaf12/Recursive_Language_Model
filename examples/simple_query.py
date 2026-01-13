#!/usr/bin/env python3
"""
Simple Query Example

Demonstrates basic RLM usage with a small context.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm import RLM


def main():
    print("=" * 60)
    print("RLM Simple Query Example")
    print("=" * 60)
    print()

    # Create a simple context
    context = """
    Company Information Report
    ==========================

    Section 1: Overview
    The company TechCorp was founded in 2015 by Alice Johnson.
    It specializes in artificial intelligence solutions.

    Section 2: Financials
    Annual revenue: $50 million (2024)
    Number of employees: 250
    Headquarters: San Francisco, CA

    Section 3: Products
    - AI Assistant Pro: Enterprise chatbot solution
    - DataMiner X: Data analytics platform
    - SecureVault: Cloud security service

    Section 4: Security
    The master access code for the admin panel is: ALPHA-7749
    This code should only be shared with authorized personnel.

    Section 5: Contact
    Email: info@techcorp.example
    Phone: +1-555-0123
    """

    query = "What is the master access code for the admin panel?"

    print(f"Context length: {len(context):,} characters")
    print(f"Query: {query}")
    print()
    print("Running RLM...")
    print("-" * 60)

    # Initialize RLM with verbose output
    rlm = RLM(verbose=True)

    # Run the query
    try:
        answer = rlm.query(query, context)

        print()
        print("=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(answer)
        print()

        # Show statistics
        stats = rlm.get_stats()
        print("Statistics:")
        print(f"  - Iterations: {stats['last_iteration_count']}")
        print(f"  - Sub-LLM calls: {stats['last_sub_llm_calls']}")
        print(f"  - Total LLM calls: {stats['client_stats']['total_calls']}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
