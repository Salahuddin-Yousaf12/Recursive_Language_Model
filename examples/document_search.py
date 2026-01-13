#!/usr/bin/env python3
"""
Document Search Example

Demonstrates RLM with multiple documents, similar to the
BrowseComp+ benchmark from the paper.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm import RLM


def generate_sample_documents(num_docs: int = 20) -> str:
    """Generate sample documents for testing"""

    documents = []

    # Add some filler documents
    topics = [
        ("Weather Report", "Today's weather shows partly cloudy skies with temperatures around 72F."),
        ("Sports News", "The local team won their game 3-2 in overtime last night."),
        ("Recipe", "To make pasta, boil water, add salt, cook for 8-10 minutes."),
        ("Technology", "New smartphones feature improved cameras and longer battery life."),
        ("Health Tips", "Regular exercise and balanced diet are key to good health."),
        ("Travel Guide", "Paris is known for the Eiffel Tower and excellent cuisine."),
        ("Book Review", "The latest bestseller explores themes of adventure and discovery."),
        ("Movie News", "The upcoming blockbuster features an all-star cast."),
        ("Science", "Recent discoveries show new species in the deep ocean."),
        ("Business", "Stock markets showed moderate gains this quarter."),
    ]

    # Generate filler documents
    for i in range(num_docs - 2):
        topic, content = topics[i % len(topics)]
        documents.append(f"""
=== Document {i + 1}: {topic} ===
Date: 2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}
Category: {topic}

{content}

This is additional filler content to make the document longer.
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
The document ID is DOC-{i + 1:04d}.
""")

    # Add the target document with the answer (somewhere in the middle)
    target_position = num_docs // 2
    documents.insert(target_position, """
=== Document SPECIAL: Festival Information ===
Date: 2024-03-15
Category: Culture

The Dinengdeng Festival is celebrated annually in Agoo, La Union, Philippines.
This vegetable stew festival has been running for 13 years now.

During the 13th anniversary celebration in 2017, several competitions were held.
The Festival of Festivals competition showcased provincial festivities.

A beauty pageant was also part of the celebration.
The winner of Miss Agoo 2017 was Maria Camille Dalmacio.

The festival celebrates the local cuisine and brings the community together.
""")

    # Add another relevant document
    documents.append("""
=== Document INFO: Philippine Festivals ===
Date: 2024-01-10
Category: Reference

Philippine festivals are known for their vibrant celebrations.
La Union province hosts several annual festivals including:
- Dinengdeng Festival in Agoo
- Surfing competitions in San Juan
- Various town fiestas

The Dinengdeng Festival started in 2004 and features the famous vegetable stew
made with fish and bagoong (fermented fish paste).
""")

    return "\n".join(documents)


def main():
    print("=" * 60)
    print("RLM Document Search Example")
    print("=" * 60)
    print()

    # Generate sample documents
    num_docs = 20
    context = generate_sample_documents(num_docs)

    print(f"Generated {num_docs} documents")
    print(f"Total context length: {len(context):,} characters")
    print()

    # Complex multi-hop query (similar to BrowseComp+ from the paper)
    query = """
    A township holds a celebration named after a vegetable stew.
    During its 13th anniversary, a beauty pageant was held.
    What is the full name of the person who won that beauty pageant?
    """

    print(f"Query: {query.strip()}")
    print()
    print("Running RLM...")
    print("-" * 60)

    # Initialize RLM
    rlm = RLM(verbose=True, max_iterations=15)

    # Run the query
    try:
        answer = rlm.query(query, context)

        print()
        print("=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(answer)
        print()

        # Expected answer: Maria Camille Dalmacio
        print("Expected answer: Maria Camille Dalmacio")
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
