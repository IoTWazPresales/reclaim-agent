"""Main entry point for Reclaim Agent."""

import sys
import os
from pathlib import Path
from datetime import datetime

from .config import Config
from .runner import Runner
from .summary import generate_daily_summary


def main():
    """Main entry point."""
    print("Reclaim Agent starting...")
    
    # Load config
    try:
        config = Config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)
    
    # Validate required settings
    if not config.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    if not config.github_token:
        print("ERROR: RECLAIM_GH_TOKEN not set")
        sys.exit(1)
    
    if not config.repo_path or not Path(config.repo_path).exists():
        print(f"ERROR: RECLAIM_REPO_PATH not set or invalid: {config.repo_path}")
        sys.exit(1)
    
    print(f"Repository: {config.repo_name}")
    print(f"Repository path: {config.repo_path}")
    print(f"Default branch: {config.default_branch}")
    
    # Run agent
    runner = Runner(config)
    
    # Determine mode
    mode = os.getenv("AGENT_MODE", "auto")
    print(f"Mode: {mode}")
    
    try:
        pr_url = runner.run(mode=mode)
        
        if pr_url:
            print(f"Agent completed successfully - PR: {pr_url}")
        else:
            print("Agent completed - no work needed")
        
        # Generate daily summary (only once per day, at end of day)
        current_hour = datetime.now().hour
        if current_hour >= 16:  # After 16:00 UTC (end of SA working day)
            github = runner.github
            prs_today = [pr_url] if pr_url else []
            generate_daily_summary(
                config,
                github,
                runs_attempted=1,
                prs_created=prs_today,
                failing_checks=runner.run_truth_checks(),
            )
        
        sys.exit(0)
    except Exception as e:
        print(f"Agent failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
