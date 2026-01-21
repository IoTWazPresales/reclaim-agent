"""Daily summary generation for the agent."""

from typing import Dict, Any, List, Optional
from datetime import datetime

from .config import Config
from .github_api import GitHubAPI
from .milestones import get_milestones_by_status


def generate_daily_summary(
    config: Config,
    github: GitHubAPI,
    runs_attempted: int,
    prs_created: List[str],
    failing_checks: List[Dict[str, Any]],
    date: Optional[str] = None,
) -> None:
    """Generate and post daily summary issue."""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    title = f"Agent Daily Summary ({date})"
    health = "🔴 RED" if failing_checks else "🟢 GREEN"
    
    # Get milestone statuses
    todo_milestones = get_milestones_by_status(config.milestones, "todo")
    in_progress_milestones = get_milestones_by_status(config.milestones, "in_progress")
    done_milestones = get_milestones_by_status(config.milestones, "done")
    blocked_milestones = get_milestones_by_status(config.milestones, "blocked")
    
    # Format PR links
    pr_links = "\n".join([f"- {pr}" for pr in prs_created]) if prs_created else "- None"
    
    # Format milestones
    def format_milestone_list(milestones: List[Dict[str, Any]]) -> str:
        if not milestones:
            return "  - None"
        return "\n".join([f"  - [{m['id']}] {m['title']}" for m in milestones])
    
    blocked_details = ""
    if blocked_milestones:
        blocked_details = "\n\n**Blocked Items:**\n"
        for m in blocked_milestones:
            reason = m.get("reason", "Unknown reason")
            blocked_details += f"- [{m['id']}] {m['title']}: {reason}\n"
    
    body = f"""## Agent Daily Summary ({date})

### Overview
- **Runs Attempted**: {runs_attempted}
- **PRs Created**: {len(prs_created)}
- **Repo Health**: {health}

### Repository Health
{f"❌ {len(failing_checks)} failing checks" if failing_checks else "✅ All truth checks passing"}

### Pull Requests Created
{pr_links}

### Milestone Status

**Todo**: {len(todo_milestones)}
{format_milestone_list(todo_milestones)}

**In Progress**: {len(in_progress_milestones)}
{format_milestone_list(in_progress_milestones)}

**Done**: {len(done_milestones)}
{format_milestone_list(done_milestones)}

**Blocked**: {len(blocked_milestones)}
{format_milestone_list(blocked_milestones)}
{blocked_details}

---
*Generated automatically by Reclaim Agent*
"""
    
    # Create or update issue
    issue = github.create_or_update_issue(title, body)
    
    if issue:
        print(f"Daily summary created/updated: {issue['html_url']}")
    else:
        print("Failed to create/update daily summary")
