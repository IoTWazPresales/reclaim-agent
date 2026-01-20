"""Prompt templates for LLM interactions.

Defensive OpenAI caller:
- Uses Responses API (v1/responses).
- Extracts text from multiple payload shapes (including payload["text"]).
- Retries automatically when the response is incomplete due to max_output_tokens.
- Avoids sending unsupported parameters unless explicitly configured.
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
    Extract primary text from Responses API payload.

    We support:
    - payload["output_text"] (convenience)
    - payload["text"]["value"] / payload["text"]["content"] (common streaming container)
    - payload["output"][...]["content"][...]["text"] (message content blocks)
    - payload["output"][...]["text"] (direct text chunks)
    """
    # 1) Convenience field
    ot = payload.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    # 2) Common top-level text container (your log shows "text": {"format": ...})
    t = payload.get("text")
    if isinstance(t, dict):
        # Some variants use "value"
        v = t.get("value")
        if isinstance(v, str) and v.strip():
            return v.strip()
        # Others use "content"
        c = t.get("content")
        if isinstance(c, str) and c.strip():
            return c.strip()
        # Some use a nested array
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

    # 3) output list parsing
    output = payload.get("output")
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue

            # Direct chunk
            if item.get("type") in ("output_text", "text") and isinstance(item.get("text"), str):
                txt = item["text"].strip()
                if txt:
                    chunks.append(txt)
                continue

            # Message content array
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


def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """
    Call OpenAI Responses API and return unified diff text.

    Key reliability feature:
    - If response.status == "incomplete" due to "max_output_tokens", retry once with higher token budget.
    """
    import requests

    debug = _env_flag("AGENT_DEBUG")

    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1"
    base = os.getenv("OPENAI_ENDPOINT", "").strip() or "https://api.openai.com"
    url = f"{base}/v1/responses"

    # Timeouts
    timeout_s = 90
    t = os.getenv("OPENAI_TIMEOUT_S", "").strip()
    if t:
        try:
            timeout_s = int(t)
        except Exception:
            timeout_s = 90

    # Initial output token budget
    max_out = 4000
    mo = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
    if mo:
        try:
            max_out = int(mo)
        except Exception:
            max_out = 4000

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Base request payload builder
    def make_data(max_output_tokens: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": "You are a code modification agent that outputs only unified diff patches."},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,
        }

        # Only include temperature if explicitly set (some models reject it)
        temp = os.getenv("OPENAI_TEMPERATURE", "").strip()
        if temp:
            try:
                data["temperature"] = float(temp)
            except ValueError:
                pass

        # If you want to try to reduce reasoning overhead, uncomment this.
        # Some model families honor it; others ignore it.
        # data["reasoning"] = {"effort": "low"}

        return data

    retries = 3
    backoff_s = 2

    # We will allow ONE “incomplete due to tokens” retry with a bigger budget
    # (within a safe cap so you don’t explode costs).
    bumped_once = False
    bump_cap = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS_CAP", "").strip() or "20000")

    for attempt in range(1, retries + 1):
        try:
            data = make_data(max_out)

            if debug:
                print(f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} attempt={attempt}/{retries} max_output_tokens={max_out}")

            response = requests.post(url, headers=headers, json=data, timeout=timeout_s)
            status_code = response.status_code

            if status_code == 200:
                raw = response.text
                payload = response.json()

                extracted = _extract_text_from_responses_api(payload)

                if debug:
                    preview = _safe_trunc(extracted.replace("\n", "\\n"), 400) if extracted else ""
                    print(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {preview}")
                    if not extracted:
                        print(f"[AGENT_DEBUG] OpenAI raw response (first 2000 chars): {_safe_trunc(raw, 2000)}")

                # If we got text, return it
                if extracted and extracted.strip():
                    return extracted.strip()

                # If incomplete due to max_output_tokens, bump and retry once immediately.
                resp_status = payload.get("status")
                inc = payload.get("incomplete_details") or {}
                reason = inc.get("reason")

                if (resp_status == "incomplete" and reason == "max_output_tokens" and not bumped_once):
                    bumped_once = True
                    # Heuristic: 4000 -> 12000 (or x3), capped
                    new_budget = min(max_out * 3, bump_cap)
                    if debug:
                        print(f"[AGENT_DEBUG] Response incomplete due to max_output_tokens. Bumping max_output_tokens {max_out} -> {new_budget} and retrying once.")
                    max_out = new_budget
                    # Don’t consume the attempt; just continue loop (same attempt count OK).
                    continue

                return None

            # Non-200
            err_json = None
            if debug:
                print(f"[AGENT_DEBUG] OpenAI non-200 status={status_code}")
                try:
                    err_json = response.json()
                    print(f"[AGENT_DEBUG] OpenAI error JSON: {err_json}")
                except Exception:
                    print(f"[AGENT_DEBUG] OpenAI error text: {_safe_trunc(response.text, 2000)}")

            # Parse error code
            if err_json is None:
                try:
                    err_json = response.json()
                except Exception:
                    err_json = None

            err_code = None
            if isinstance(err_json, dict):
                err = err_json.get("error") or {}
                err_code = err.get("code") or err.get("type")

            if err_code == "insufficient_quota":
                print("OpenAI quota exhausted / billing not active for this API key (insufficient_quota).")
                return None

            # Retry transient
            if status_code in (429, 500, 502, 503, 504) and attempt < retries:
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