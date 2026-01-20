"""Prompt templates for LLM interactions."""

from typing import List, Dict, Any, Optional
import os
import time


def build_fix_prompt(
    failing_checks: List[Dict[str, Any]],
    repo_rules: List[str],
    max_files: int,
    max_lines: int
) -> str:
    """Build prompt for fix mode."""
    check_details = "\n".join([
        f"- {check['name']}: {check.get('error', 'Failed')}"
        for check in failing_checks
    ])

    rules_text = "\n".join([f"- {rule}" for rule in repo_rules])

    return f"""You are a code fixing agent for the Reclaim repository. Fix the failing truth checks below.

REPO RULES (CRITICAL - MUST FOLLOW):
{rules_text}

FAILING CHECKS:
{check_details}

CONSTRAINTS:
- Maximum {max_files} files changed
- Maximum {max_lines} lines net change (additions - deletions)
- Output ONLY a unified diff patch in the following format:
  - Start with `--- a/path/to/file`
  - Follow with `+++ b/path/to/file`
  - Include context lines before/after changes
  - Use standard unified diff format

OUTPUT FORMAT:
Provide a unified diff patch ONLY. Do not include explanations, comments, or markdown formatting.
Start immediately with the diff:

--- a/path/to/file
+++ b/path/to/file
@@ -line,count +line,count @@
 context line
-old line
+new line
 context line
"""


def build_milestone_prompt(
    milestone: Dict[str, Any],
    repo_rules: List[str],
    max_files: int,
    max_lines: int,
    current_files: Optional[str] = None
) -> str:
    """Build prompt for milestone mode."""
    acceptance = "\n".join([f"- {cmd}" for cmd in milestone.get("acceptance", [])])
    target_files = milestone.get("target_files", [])
    files_context = ""
    if target_files:
        files_context = (
            f"\nTARGET FILES (focus on these patterns):\n"
            + "\n".join([f"- {pattern}" for pattern in target_files])
        )

    if current_files:
        files_context += f"\n\nCURRENT FILE CONTENTS (if relevant):\n{current_files}"

    rules_text = "\n".join([f"- {rule}" for rule in repo_rules])

    return f"""You are a code modification agent for the Reclaim repository. Complete the milestone task below.

REPO RULES (CRITICAL - MUST FOLLOW):
{rules_text}

MILESTONE:
Title: {milestone['title']}
Type: {milestone['type']}
Acceptance Criteria (all must pass):
{acceptance}
{files_context}

CONSTRAINTS:
- Maximum {max_files} files changed
- Maximum {max_lines} lines net change (additions - deletions)
- Output ONLY a unified diff patch in the following format:
  - Start with `--- a/path/to/file`
  - Follow with `+++ b/path/to/file`
  - Include context lines before/after changes
  - Use standard unified diff format

OUTPUT FORMAT:
Provide a unified diff patch ONLY. Do not include explanations, comments, or markdown formatting.
Start immediately with the diff:

--- a/path/to/file
+++ b/path/to/file
@@ -line,count +line,count @@
 context line
-old line
+new line
 context line
"""


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """Call OpenAI API with prompt and return response content (unified diff)."""
    import requests

    debug = _env_flag("AGENT_DEBUG")

    # Prefer env override so you can change without code edits.
    # Safe default for broad availability on Chat Completions:
    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a code modification agent that outputs only unified diff patches."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4000,
    }

    # Small retry for transient failures (rate limit / 5xx)
    retries = 3
    backoff_s = 2

    for attempt in range(1, retries + 1):
        try:
            if debug:
                print(f"[AGENT_DEBUG] OpenAI request: model={model} attempt={attempt}/{retries}")

            response = requests.post(url, headers=headers, json=data, timeout=60)
            status = response.status_code

            # Success path
            if status == 200:
                result = response.json()
                # Defensive extraction
                choices = result.get("choices") or []
                if not choices:
                    if debug:
                        print("[AGENT_DEBUG] OpenAI response had no choices.")
                        print(f"[AGENT_DEBUG] Raw JSON keys: {list(result.keys())}")
                    return None

                msg = (choices[0].get("message") or {})
                content = (msg.get("content") or "").strip()

                if debug:
                    preview = content[:400].replace("\n", "\\n")
                    print(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {preview}")

                return content or None

            # Non-200: log useful error
            if debug:
                print(f"[AGENT_DEBUG] OpenAI non-200 status={status}")
                try:
                    err_json = response.json()
                    print(f"[AGENT_DEBUG] OpenAI error JSON: {err_json}")
                except Exception:
                    print(f"[AGENT_DEBUG] OpenAI error text: {response.text[:2000]}")

            # Retry on rate limit / server errors
            if status in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue

            return None

        except Exception as e:
            if debug:
                print(f"[AGENT_DEBUG] OpenAI API exception: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            return None

    return None
