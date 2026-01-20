"""Prompt templates for LLM interactions.

Defensive OpenAI caller:
- Uses Responses API (v1/responses).
- Extracts text from multiple payload shapes.
- Retries automatically when response is incomplete due to max_output_tokens.
- Uses adaptive timeouts for larger outputs.
- Strips unsupported parameters on 400 "unsupported_value"/"unsupported_parameter" and retries once.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import os
import time


def build_fix_prompt(
    failing_checks: List[Dict[str, Any]],
    repo_rules: List[str],
    max_files: int,
    max_lines: int
) -> str:
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

CRITICAL OUTPUT REQUIREMENT:
- You MUST output ONLY a unified diff patch.
- DO NOT output analysis, reasoning, explanations, or markdown fences.
- Start your very first character with `--- a/` (the diff header).

CONSTRAINTS:
- Maximum {max_files} files changed
- Maximum {max_lines} lines net change (additions - deletions)

OUTPUT FORMAT:
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
    """Extract primary text from Responses API payload."""
    ot = payload.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    t = payload.get("text")
    if isinstance(t, dict):
        v = t.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()
        c = t.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        arr = t.get("chunks") or t.get("parts")
        if isinstance(arr, list):
            chunks = []
            for item in arr:
                if isinstance(item, str) and item.strip():
                    chunks.append(item.strip())
                elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                    chunks.append(item["text"].strip())
            joined = "\n".join(chunks).strip()
            if joined:
                return joined

    output = payload.get("output")
    if isinstance(output, list):
        chunks = []
        for item in output:
            if not isinstance(item, dict):
                continue

            if item.get("type") in ("output_text", "text") and isinstance(item.get("text"), str):
                txt = item["text"].strip()
                if txt:
                    chunks.append(txt)
                continue

            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("output_text", "text") and isinstance(part.get("text"), str):
                        txt = part["text"].strip()
                        if txt:
                            chunks.append(txt)

        joined = "\n".join(chunks).strip()
        if joined:
            return joined

    return ""


def _parse_openai_error(resp_json: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return (code/type, param, message) from an OpenAI-style error response JSON.
    """
    if not isinstance(resp_json, dict):
        return None, None, None
    err = resp_json.get("error")
    if not isinstance(err, dict):
        return None, None, None
    code_or_type = err.get("code") or err.get("type")
    param = err.get("param")
    msg = err.get("message")
    return code_or_type, param, msg


def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """
    Call OpenAI Responses API and return unified diff text.

    Reliability features:
    - Adaptive timeouts (bigger max_output_tokens => bigger read timeout).
    - Gradual max_output_tokens bump on truncation.
    - If a 400 reports an unsupported param/value, strip it and retry once.
    """
    import requests

    debug = _env_flag("AGENT_DEBUG")

    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1"
    base = os.getenv("OPENAI_ENDPOINT", "").strip() or "https://api.openai.com"
    url = f"{base}/v1/responses"

    # Separate connect/read timeouts
    connect_timeout_s = int(os.getenv("OPENAI_CONNECT_TIMEOUT_S", "15"))
    base_read_timeout_s = int(os.getenv("OPENAI_READ_TIMEOUT_S", "120"))

    # Initial output token budget
    max_out = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4000"))
    bump_cap = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS_CAP", "20000"))

    # Default bump steps (you can override with "8000,12000,16000")
    bump_steps_env = os.getenv("OPENAI_BUMP_STEPS", "").strip()
    if bump_steps_env:
        try:
            bump_steps = [int(x.strip()) for x in bump_steps_env.split(",") if x.strip()]
        except Exception:
            bump_steps = [8000, 16000]
    else:
        bump_steps = [8000, 16000]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # We will only do ONE "strip unsupported params" retry per outer attempt
    stripped_once = False

    def make_data(max_output_tokens: int, include_reasoning: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": "Output ONLY a unified diff patch. Start immediately with `--- a/`."},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,
        }

        # Only include reasoning control if enabled (some models may reject it)
        if include_reasoning:
            data["reasoning"] = {"effort": "low"}

        # IMPORTANT: Do NOT send text.verbosity; codex shows only "medium" supported in your logs.
        # If later you want to set it, set it to "medium" explicitly via env and only include if set.
        tv = os.getenv("OPENAI_TEXT_VERBOSITY", "").strip()
        if tv:
            data["text"] = {"verbosity": tv}

        temp = os.getenv("OPENAI_TEMPERATURE", "").strip()
        if temp:
            try:
                data["temperature"] = float(temp)
            except ValueError:
                pass

        return data

    def adaptive_timeout(max_output_tokens: int) -> Tuple[int, int]:
        read = base_read_timeout_s
        if max_output_tokens >= 8000:
            read = max(read, 180)
        if max_output_tokens >= 16000:
            read = max(read, 300)
        return connect_timeout_s, read

    retries = 3
    backoff_s = 2
    bump_index = -1
    include_reasoning = True

    for attempt in range(1, retries + 1):
        try:
            timeout = adaptive_timeout(max_out)
            data = make_data(max_out, include_reasoning=include_reasoning)

            if debug:
                print(
                    f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} "
                    f"attempt={attempt}/{retries} max_output_tokens={max_out} timeout={timeout} "
                    f"include_reasoning={include_reasoning}"
                )

            resp = requests.post(url, headers=headers, json=data, timeout=timeout)

            if resp.status_code == 200:
                payload = resp.json()
                extracted = _extract_text_from_responses_api(payload)

                if debug:
                    preview = _safe_trunc(extracted.replace("\n", "\\n"), 400) if extracted else ""
                    print(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {preview}")
                    if not extracted:
                        print(f"[AGENT_DEBUG] OpenAI raw response (first 2000 chars): {_safe_trunc(resp.text, 2000)}")

                if extracted and extracted.strip():
                    return extracted.strip()

                # Incomplete due to max_output_tokens: bump gradually
                resp_status = payload.get("status")
                inc = payload.get("incomplete_details") or {}
                reason = inc.get("reason")

                if resp_status == "incomplete" and reason == "max_output_tokens":
                    next_budget = None
                    if bump_index + 1 < len(bump_steps):
                        bump_index += 1
                        next_budget = bump_steps[bump_index]
                    else:
                        next_budget = min(max_out * 2, bump_cap)

                    next_budget = min(next_budget, bump_cap)

                    if next_budget > max_out:
                        if debug:
                            print(f"[AGENT_DEBUG] Incomplete due to max_output_tokens. Bumping {max_out} -> {next_budget} and retrying.")
                        max_out = next_budget
                        continue

                return None

            # Non-200
            if debug:
                print(f"[AGENT_DEBUG] OpenAI non-200 status={resp.status_code}")
                try:
                    print(f"[AGENT_DEBUG] OpenAI error JSON: {resp.json()}")
                except Exception:
                    print(f"[AGENT_DEBUG] OpenAI error text: {_safe_trunc(resp.text, 2000)}")

            # Try to strip unsupported params/values and retry once immediately
            if resp.status_code == 400 and not stripped_once:
                stripped_once = True
                try:
                    err_json = resp.json()
                except Exception:
                    err_json = None

                code_or_type, param, msg = _parse_openai_error(err_json or {})
                msg_str = str(msg or "")

                # Cases we've seen:
                # - Unsupported parameter: temperature
                # - Unsupported value: text.verbosity=low
                if code_or_type in ("invalid_request_error", "unsupported_value", "unsupported_parameter"):
                    if debug:
                        print(f"[AGENT_DEBUG] 400 indicates unsupported setting (param={param}) message={msg_str}")

                    # If reasoning control is being rejected, disable it and retry.
                    if "reasoning" in msg_str or (param and str(param).startswith("reasoning")):
                        include_reasoning = False
                        if debug:
                            print("[AGENT_DEBUG] Disabling reasoning control and retrying once.")
                        continue

                    # If text.verbosity is rejected, remove it by clearing env usage (we don't send unless env set).
                    if "text.verbosity" in msg_str or (param == "text.verbosity"):
                        # Ensure we are not sending it
                        os.environ.pop("OPENAI_TEXT_VERBOSITY", None)
                        if debug:
                            print("[AGENT_DEBUG] Removing text.verbosity and retrying once.")
                        continue

                    # If temperature rejected, remove it (we only send if env set)
                    if "temperature" in msg_str or (param == "temperature"):
                        os.environ.pop("OPENAI_TEMPERATURE", None)
                        if debug:
                            print("[AGENT_DEBUG] Removing temperature and retrying once.")
                        continue

                # Otherwise: fall through (no retry)
                return None

            # Retry transient codes
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue

            return None

        except requests.exceptions.ReadTimeout as e:
            if debug:
                print(f"[AGENT_DEBUG] OpenAI ReadTimeout: {e}")
            # Increase base read timeout for subsequent attempts
            base_read_timeout_s = max(base_read_timeout_s, 300)

            if attempt < retries:
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