"""Command-line script to generate knowledge base."""

import sys
import os
from pathlib import Path
from .knowledge_base import KnowledgeBaseGenerator


def main():
    """Generate knowledge base for Reclaim repository."""
    repo_path = os.getenv("RECLAIM_REPO_PATH", "")
    
    if not repo_path or not Path(repo_path).exists():
        print(f"ERROR: RECLAIM_REPO_PATH not set or invalid: {repo_path}")
        print("Usage: Set RECLAIM_REPO_PATH environment variable to the Reclaim repository path")
        sys.exit(1)
    
    print(f"Generating knowledge base for: {repo_path}")
    
    generator = KnowledgeBaseGenerator(repo_path)
    output_path = generator.save()
    
    print(f"Knowledge base generated: {output_path}")
    print(f"Size: {len(output_path.read_text(encoding='utf-8'))} characters")
    sys.exit(0)


if __name__ == "__main__":
    main()
