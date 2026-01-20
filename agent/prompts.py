"""Prompt templates + OpenAI call wrapper for LLM interactions.

This module is designed to be robust for GitHub Actions usage:
- Uses OpenAI Responses API (/v1/responses) to support newer models.
- Avoids unsupported parameters for codex-style models.
- Adds env-configurable timeouts + token limits.
- Provides conservative output extraction for unified-diff-only responses.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import time
import json


# ---------------------------
# Prompt builders
# ---------------------------

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

    # Keep this extremely direct to minimize tokens wasted on planning.
    return f"""You are a code fixing agent for the Reclaim repository. Fix the failing truth checks below.

REPO RULES (CRITICAL - MUST FOLLOW):
{rules_text}

FAILING CHECKS:
{check_details}

CONSTRAINTS:
- Maximum {max_files} files changed
- Maximum {max_lines} lines net change (additions - deletions)
- Output ONLY a unified diff patch.

OUTPUT FORMAT (STRICT):
- Output ONLY a unified diff patch.
- NO explanations, NO markdown fences, NO commentary.
- Start immediately with:  --- a/...

Begin now:
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
            "\nTARGET FILES (focus on these patterns):\n"
            + "\n".join([f"- {pattern}" for pattern in target_files])
        )

    if current_files:
        files_context += f"\n\nCURRENT FILE CONTENTS (if relevant):\n{current_files}"

    rules_text = "\n".join([f"- {rule}" for rule in repo_rules])

    # Keep short + strict to reduce “reasoning-only” output and token burn.
    return f"""You are a code modification agent for the Reclaim repository. Complete the milestone below.

REPO RULES (CRITICAL - MUST FOLLOW):
{rules_text}

MILESTONE:
Title: {milestone['title']}
Type: {milestone.get('type', 'feat')}
Acceptance commands (all must pass):
{acceptance}
{files_context}

CONSTRAINTS:
- Maximum {max_files} files changed
- Maximum {max_lines} lines net change (additions - deletions)

OUTPUT FORMAT (STRICT):
- Output ONLY a unified diff patch.
- NO explanations, NO markdown fences, NO commentary.
- Start immediately with:  --- a/...

Begin now:
"""


# ---------------------------
# Helpers / env
# ---------------------------

def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _extract_text_from_responses_api(payload: dict) -> str:
    """
    Extract the primary text output from OpenAI Responses API JSON.

    Expected shapes:
    - payload["output_text"] (sometimes present)
    - payload["output"] = [ { "type": "...", "content": [ { "type": "output_text", "text": "..." } ] } ]
    """
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    out = payload.get("output")
    if isinstance(out, list):
        parts: list[str] = []
        for item in out:
            # We only care about message/text content, ignore reasoning blocks.
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    ctype = c.get("type")
                    if ctype in ("output_text", "text") and isinstance(c.get("text"), str):
                        parts.append(c["text"])
        text = "\n".join(parts).strip()
        if text:
            return text

    return ""


# ---------------------------
# OpenAI call
# ---------------------------

def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """
    Call OpenAI Responses API with prompt and return the response text (expected: unified diff).
    Returns None on failure.
    """
    import requests

    debug = _env_flag("AGENT_DEBUG")

    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1"
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Timeouts
    connect_timeout_s = _env_int("OPENAI_CONNECT_TIMEOUT_S", 15)
    read_timeout_s = _env_int("OPENAI_READ_TIMEOUT_S", 120)

    # Token controls
    start_tokens = _env_int("OPENAI_MAX_OUTPUT_TOKENS", 4000)
    cap_tokens = _env_int("OPENAI_MAX_OUTPUT_TOKENS_CAP", 20000)
    max_output_tokens = _clamp(start_tokens, 512, cap_tokens)

    # Retries
    retries = _env_int("OPENAI_RETRIES", 3)
    backoff_s = _env_int("OPENAI_BACKOFF_S", 2)

    # Build base request body (Responses API)
    data: Dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": "You output ONLY unified diff patches. Start immediately with '--- a/...'. No explanations."},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": max_output_tokens,
        # IMPORTANT: codex-style models in your logs only support text.verbosity="medium".
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        # Do not add reasoning controls unless you are 100% sure the model supports them.
    }

    # Optional temperature (ONLY if explicitly set)
    temp = _env_float("OPENAI_TEMPERATURE")
    if temp is not None:
        data["temperature"] = temp

    def _dbg(msg: str) -> None:
        if debug:
            print(msg)

    for attempt in range(1, retries + 1):
        try:
            _dbg(f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} attempt={attempt}/{retries} max_output_tokens={data.get('max_output_tokens')} timeout=({connect_timeout_s}, {read_timeout_s})")

            resp = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=(connect_timeout_s, read_timeout_s),
            )

            status = resp.status_code

            if status == 200:
                payload = resp.json()

                # Debug raw if needed
                if debug:
                    raw_preview = json.dumps(payload, ensure_ascii=False)[:2000]
                    print(f"[AGENT_DEBUG] OpenAI raw response (first 2000 chars): {raw_preview}")

                content = _extract_text_from_responses_api(payload)

                _dbg(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {(content[:400] if content else '').replace(chr(10), '\\n')}")

                # Handle incomplete due to token cap (very common in your logs)
                if (not content) and isinstance(payload, dict) and payload.get("status") == "incomplete":
                    inc = payload.get("incomplete_details") or {}
                    reason = inc.get("reason")
                    if reason == "max_output_tokens":
                        # bump once and retry immediately
                        current = int(data.get("max_output_tokens") or max_output_tokens)
                        bumped = _clamp(max(current * 3, current + 2000), 512, cap_tokens)
                        if bumped > current:
                            _dbg(f"[AGENT_DEBUG] Response incomplete due to max_output_tokens. Bumping {current} -> {bumped} and retrying.")
                            data["max_output_tokens"] = bumped
                            # increase read timeout a bit on next attempt
                            read_timeout_s = max(read_timeout_s, _env_int("OPENAI_READ_TIMEOUT_S", 120))
                            if attempt < retries:
                                continue

                return content.strip() if content and content.strip() else None

            # Non-200
            _dbg(f"[AGENT_DEBUG] OpenAI non-200 status={status}")

            err_json = None
            try:
                err_json = resp.json()
                _dbg(f"[AGENT_DEBUG] OpenAI error JSON: {err_json}")
            except Exception:
                _dbg(f"[AGENT_DEBUG] OpenAI error text: {resp.text[:2000]}")

            # If we hit the exact verbosity issue again, force medium / remove field then retry once.
            if isinstance(err_json, dict):
                err = err_json.get("error") or {}
                param = err.get("param")
                msg = (err.get("message") or "").lower()

                if param == "text.verbosity" or "verbosity" in msg:
                    # codex supports only medium (per your logs). Hard-force it.
                    data["text"] = {"format": {"type": "text"}, "verbosity": "medium"}
                    if attempt < retries:
                        time.sleep(backoff_s)
                        backoff_s = min(backoff_s * 2, 30)
                        continue

                if err.get("code") == "insufficient_quota" or err.get("type") == "insufficient_quota":
                    print("OpenAI quota exhausted / billing not active for this API key (insufficient_quota).")
                    return None

            # Retry on transient statuses
            if status in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30)
                continue

            return None

        except requests.exceptions.ReadTimeout as e:
            _dbg(f"[AGENT_DEBUG] OpenAI ReadTimeout: {e}")
            if attempt < retries:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30)
                continue
            return None

        except Exception as e:
            _dbg(f"[AGENT_DEBUG] OpenAI API exception: {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30)
                continue
            return None

    return None