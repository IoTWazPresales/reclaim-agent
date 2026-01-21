"""Command-line script to generate knowledge base."""

import sys
import os
from pathlib import Path
from .knowledge_base import KnowledgeBaseGenerator


def main():
    """Generate knowledge base for Reclaim repository."""
    repo_path = os.getenv("RECLAIM_REPO_PATH", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_llm = os.getenv("KB_USE_LLM", "1").strip().lower() in ("1", "true", "yes")
    
    if not repo_path or not Path(repo_path).exists():
        print(f"ERROR: RECLAIM_REPO_PATH not set or invalid: {repo_path}")
        print("Usage: Set RECLAIM_REPO_PATH environment variable to the Reclaim repository path")
        sys.exit(1)
    
    print(f"Generating knowledge base for: {repo_path}")
    
    if use_llm and api_key:
        print("Using hybrid approach: Python structure + LLM semantic analysis (gpt-4o-mini)")
    else:
        print("Using structure-only approach (no LLM, zero tokens)")
        if use_llm and not api_key:
            print("  (OPENAI_API_KEY not set, skipping LLM analysis)")
    
    generator = KnowledgeBaseGenerator(
        repo_path,
        openai_api_key=api_key if use_llm else None,
        use_llm_analysis=use_llm and bool(api_key)
    )
    output_path = generator.save()
    
    kb_size = len(output_path.read_text(encoding='utf-8'))
    print(f"Knowledge base generated: {output_path}")
    print(f"Size: {kb_size:,} characters")
    sys.exit(0)


if __name__ == "__main__":
    main()
