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


def _extract_text_from_responses_api(payload: dict) -> str:
    """
    Extracts the primary text output from OpenAI Responses API JSON.
    Supports common response shapes.
    """
    # Preferred: output_text convenience field (some SDKs provide; API often does not)
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            # item can have "content": [{ "type": "output_text", "text": "..."}, ...]
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        chunks.append(c["text"])
        text = "\n".join(chunks).strip()
        if text:
            return text

    # Fallback: some variants return { "choices": ... } or other formats
    # but we keep it conservative: stringify if nothing else.
    return ""


def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """
    Call OpenAI API with prompt and return response content (unified diff).

    Uses the Responses API (v1/responses) which supports GPT-5.x / Codex-style models.
    """
    import requests

    debug = _env_flag("AGENT_DEBUG")

    # Allow override via workflow env var
    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1"

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Responses API format
    data = {
    "model": model,
    "input": [
        {"role": "system", "content": "You are a code modification agent that outputs only unified diff patches."},
        {"role": "user", "content": prompt},
    ],
    "max_output_tokens": 4000,
}

# Some models (incl. codex-style) reject temperature. Only include if explicitly set.
temp = os.getenv("OPENAI_TEMPERATURE", "").strip()
if temp:
    try:
        data["temperature"] = float(temp)
    except ValueError:
        # ignore invalid env values
        pass

    retries = 3
    backoff_s = 2

    for attempt in range(1, retries + 1):
        try:
            if debug:
                print(f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} attempt={attempt}/{retries}")

            response = requests.post(url, headers=headers, json=data, timeout=90)
            status = response.status_code

            if status == 200:
                payload = response.json()
                content = _extract_text_from_responses_api(payload)

                if debug:
                    preview = (content[:400] if content else "").replace("\n", "\\n")
                    print(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {preview}")

                return content.strip() if content and content.strip() else None

            # Non-200: log useful error
            err_json = None
            if debug:
                print(f"[AGENT_DEBUG] OpenAI non-200 status={status}")
                try:
                    err_json = response.json()
                    print(f"[AGENT_DEBUG] OpenAI error JSON: {err_json}")
                except Exception:
                    print(f"[AGENT_DEBUG] OpenAI error text: {response.text[:2000]}")

            # Extract error code if present
            if err_json is None:
                try:
                    err_json = response.json()
                except Exception:
                    err_json = None

            err_code = None
            if isinstance(err_json, dict):
                err = err_json.get("error") or {}
                err_code = err.get("code") or err.get("type")

            # Quota: retries are pointless
            if err_code == "insufficient_quota":
                print("OpenAI quota exhausted / billing not active for this API key (insufficient_quota).")
                return None

            # Retry on transient errors
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
