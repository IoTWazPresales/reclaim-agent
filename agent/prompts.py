"""Prompt templates for LLM interactions.

This module builds prompts for fix/milestone modes and calls the OpenAI API.
It is intentionally defensive:
- Supports multiple OpenAI response payload shapes (Responses API + legacy chat).
- Avoids unsupported params for certain models (e.g., temperature for some codex-style models).
- Adds rich debug output when AGENT_DEBUG=1, without leaking secrets.
"""

from __future__ import annotations

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

    # NOTE: The agent's runner applies git patches. To reduce "empty patch" / wrong-format issues,
    # we keep the instruction extremely explicit and repeat "unified diff only".
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


def _safe_trunc(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n] + "…"


def _extract_text_from_responses_api(payload: dict) -> str:
    """
    Extracts the primary text output from OpenAI Responses API JSON.
    Supports common response shapes across model families.

    Known shapes:
    1) {"output_text": "..."}  (convenience)
    2) {"output":[{"type":"message","content":[{"type":"output_text","text":"..."}]}]}
    3) {"output":[{"type":"message","content":[{"type":"text","text":"..."}]}]}
    4) {"output":[{"type":"output_text","text":"..."}]}  (direct chunks)
    """
    # Convenience field (some SDKs / responses)
    ot = payload.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    output = payload.get("output")
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue

            # Direct output chunk
            if item.get("type") in ("output_text", "text") and isinstance(item.get("text"), str):
                t = item["text"].strip()
                if t:
                    chunks.append(t)
                continue

            # Message-style
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    # Some payloads use type=text; some use output_text
                    if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                        t = c["text"].strip()
                        if t:
                            chunks.append(t)

        text = "\n".join(chunks).strip()
        if text:
            return text

    return ""


def _extract_text_from_chat_completions(payload: dict) -> str:
    """
    Extract text from legacy chat.completions format, if we ever fall back.
    Shape:
      {"choices":[{"message":{"content":"..."}}]}
    """
    try:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
    except Exception:
        pass
    return ""


def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """
    Call OpenAI API with prompt and return response content (expected unified diff).

    Default: Responses API (v1/responses).
    Optional fallback: legacy Chat Completions (v1/chat/completions) if explicitly enabled.

    Env vars:
      OPENAI_MODEL: model name override (default: gpt-4.1)
      OPENAI_TEMPERATURE: optional float; only sent when provided and parseable
      AGENT_DEBUG: 1 to print detailed logs
      OPENAI_ENDPOINT: override base endpoint (default: https://api.openai.com)
      OPENAI_USE_CHAT_COMPLETIONS_FALLBACK: 1 to allow fallback when Responses returns 200 but empty extract
      OPENAI_TIMEOUT_S: request timeout seconds (default: 90)
      OPENAI_MAX_OUTPUT_TOKENS: override max_output_tokens (default: 4000)
    """
    import requests

    debug = _env_flag("AGENT_DEBUG")

    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1"
    base = os.getenv("OPENAI_ENDPOINT", "").strip() or "https://api.openai.com"
    url = f"{base}/v1/responses"

    timeout_s = 90
    try:
        t = os.getenv("OPENAI_TIMEOUT_S", "").strip()
        if t:
            timeout_s = int(t)
    except Exception:
        timeout_s = 90

    max_out = 4000
    try:
        mo = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
        if mo:
            max_out = int(mo)
    except Exception:
        max_out = 4000

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Responses API format
    data: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": "You are a code modification agent that outputs only unified diff patches."},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": max_out,
    }

    # Only include temperature if explicitly set (some models reject it)
    temp = os.getenv("OPENAI_TEMPERATURE", "").strip()
    if temp:
        try:
            data["temperature"] = float(temp)
        except ValueError:
            pass

    retries = 3
    backoff_s = 2

    for attempt in range(1, retries + 1):
        try:
            if debug:
                print(f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} attempt={attempt}/{retries}")

            response = requests.post(url, headers=headers, json=data, timeout=timeout_s)
            status = response.status_code

            if status == 200:
                raw_text = response.text  # for debug if extraction fails
                payload = response.json()
                content = _extract_text_from_responses_api(payload)

                if debug:
                    preview = _safe_trunc(content.replace("\n", "\\n"), 400) if content else ""
                    print(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {preview}")
                    if not content:
                        # This is the key debug you were missing: show the actual shape.
                        print(f"[AGENT_DEBUG] OpenAI raw response (first 2000 chars): {_safe_trunc(raw_text, 2000)}")

                if content and content.strip():
                    return content.strip()

                # Optional fallback (disabled by default) to chat.completions if enabled
                if _env_flag("OPENAI_USE_CHAT_COMPLETIONS_FALLBACK"):
                    if debug:
                        print("[AGENT_DEBUG] Empty extract from responses; attempting chat.completions fallback")

                    chat_url = f"{base}/v1/chat/completions"
                    chat_data: Dict[str, Any] = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a code modification agent that outputs only unified diff patches."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": max_out,
                    }
                    # Only include temperature if explicitly set
                    if "temperature" in data:
                        chat_data["temperature"] = data["temperature"]

                    chat_resp = requests.post(chat_url, headers=headers, json=chat_data, timeout=timeout_s)
                    if chat_resp.status_code == 200:
                        chat_payload = chat_resp.json()
                        chat_text = _extract_text_from_chat_completions(chat_payload)
                        if debug:
                            chat_preview = _safe_trunc(chat_text.replace("\n", "\\n"), 400) if chat_text else ""
                            print(f"[AGENT_DEBUG] Chat fallback preview (first 400 chars): {chat_preview}")
                        return chat_text.strip() if chat_text and chat_text.strip() else None

                return None

            # Non-200: log useful error
            err_json = None
            if debug:
                print(f"[AGENT_DEBUG] OpenAI non-200 status={status}")
                try:
                    err_json = response.json()
                    print(f"[AGENT_DEBUG] OpenAI error JSON: {err_json}")
                except Exception:
                    print(f"[AGENT_DEBUG] OpenAI error text: {_safe_trunc(response.text, 2000)}")

            # Try parse error JSON if not already
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