# Reclaim Agent

A Python-based GitHub Actions agent that can fix CI failures and complete milestone tasks for the Reclaim repository.

## Features

- **FIX_MODE**: Automatically fixes failing truth checks (npm ci, tsc, vitest)
- **MILESTONE_MODE**: Completes milestone tasks from a queue
- **PR-only workflow**: Never merges to main, always creates PRs
- **Daily summaries**: Posts daily summary issues with progress
- **Surgical patches**: Small, focused changes by default (max 3 files, 150 lines)
- **Safe operations**: Never force pushes, deletes branches, or modifies critical systems

## Setup

### GitHub Secrets

Add these secrets to your GitHub repository settings:

- `OPENAI_API_KEY`: Your OpenAI API key for LLM operations
- `RECLAIM_GH_TOKEN`: GitHub personal access token with `repo` scope for the Reclaim repository

### GitHub Variables

Add these variables to your GitHub repository settings:

- `RECLAIM_REPO`: `IoTWazPresales/Reclaim`
- `RECLAIM_DEFAULT_BRANCH`: `main`

### Local Development

1. Clone the repository:
   ```bash
   git clone <this-repo>
   cd reclaim-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file:
   ```bash
   OPENAI_API_KEY=your_key_here
   RECLAIM_GH_TOKEN=your_token_here
   RECLAIM_REPO=IoTWazPresales/Reclaim
   RECLAIM_DEFAULT_BRANCH=main
   ```

4. Run locally:
   ```bash
   python -m agent.run
   ```

## Configuration

### Milestone Queue

Milestones are defined in `agent_config/reclaim.yaml`:

```yaml
milestones:
  - id: milestone-001
    title: "Fix TypeScript errors in training module"
    type: fix
    status: todo
    target_files: ["app/src/lib/training/**"]
    acceptance:
      - "cd app && npx tsc --noEmit"
      - "cd app && npx vitest run --passWithNoTests"
    stop_feature: false
    created_at: "2025-01-18"
```

**Milestone fields:**
- `id`: Unique identifier
- `title`: Human-readable description
- `type`: `fix` or `feature`
- `status`: `todo`, `in_progress`, `done`, `blocked`
- `target_files`: Optional list of file patterns to focus on
- `acceptance`: List of commands that must pass for completion
- `stop_feature`: If `true`, agent stops after this milestone completes
- `created_at`: Creation date (YYYY-MM-DD)

### Adding Milestones

1. Edit `agent_config/reclaim.yaml`
2. Add a new milestone to the `milestones` list
3. Set `status: todo`
4. Define `acceptance` criteria
5. Commit and push - the agent will pick it up on the next run

## How It Works

1. **Agent runs** (via GitHub Actions or locally)
2. **Checks mode**:
   - **FIX_MODE**: If truth checks fail, attempt to fix
   - **MILESTONE_MODE**: If milestone queue has `todo` items, attempt the first one
3. **Creates branch**: `agent/<date>-<milestone_or_fix>`
4. **LLM planning**: Uses OpenAI to generate a patch plan
5. **Applies patch**: Safely applies unified diff patch
6. **Verifies**: Runs acceptance criteria
7. **Creates PR**: Opens PR with summary, root cause, verification results
8. **Updates milestone**: Marks `in_progress` → `done` or `blocked`
9. **Daily summary**: Creates/updates daily summary issue at end of day

## Daily Summaries

The agent creates a GitHub Issue titled "Agent Daily Summary (YYYY-MM-DD)" that includes:

- Number of runs attempted
- PR links created during the day
- Milestone statuses
- Blocked items with reasons
- Current repo health (green/red based on truth checks)

## Safety Rules

The agent will **never**:
- Force push to branches
- Delete branches
- Modify Supabase migrations or database schema
- Touch auth flow unless milestone explicitly targets it
- Create PRs larger than 3 files / 150 lines unless escalation is justified

## Troubleshooting

### Agent doesn't run

- Check GitHub Actions workflow is enabled
- Verify secrets and variables are set correctly
- Check workflow schedule (runs every 2 hours 06:00-16:00 UTC Mon-Fri)

### Milestone stuck in `in_progress`

- Manually update `agent_config/reclaim.yaml` to set `status: todo` or `blocked`
- Check PR comments for error messages

### Truth checks fail after fix

- Review PR diff carefully
- Check LLM-generated patch applied correctly
- May need manual intervention for complex failures

## License

MIT
