"""Prompt templates + OpenAI call wrapper for LLM interactions.

Robust for GitHub Actions usage:
- Uses OpenAI Responses API (/v1/responses) to support newer models.
- Avoids unsupported parameters for codex-style models (temperature/text.verbosity).
- Adds env-configurable timeouts + token limits + retries.
- Extracts unified-diff-only responses conservatively and sanitizes to first diff.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
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

OUTPUT FORMAT (STRICT):
- Output ONLY a valid unified diff patch that `git apply` can parse.
- Use REAL line numbers in hunk headers: @@ -10,5 +10,8 @@ NOT @@ ... @@
- Include sufficient context lines (at least 3 before and after changes).
- NO explanations, NO markdown fences, NO commentary, NO placeholders.
- Start immediately with:  --- a/path/to/file.ext

EXAMPLE of correct format:
--- a/app/src/example.ts
+++ b/app/src/example.ts
@@ -15,7 +15,9 @@ export function example() {{
   const x = 1;
   const y = 2;
+  const z = 3;
   return x + y;
 }}

Begin now:
"""


def build_milestone_prompt(
    milestone: Dict[str, Any],
    repo_rules: List[str],
    max_files: int,
    max_lines: int,
    current_files: Optional[str] = None
) -> str:
    acceptance = "\n".join([f"- {cmd}" for cmd in milestone.get("acceptance", [])])

    # Optional rich spec block if present in milestone config
    spec_block = ""
    spec = milestone.get("spec")
    if spec:
        try:
            spec_text = json.dumps(spec, indent=2, ensure_ascii=False)
        except Exception:
            spec_text = str(spec)
        spec_block = f"\n\nDETAILED SPEC (authoritative for behavior, UX, and constraints):\n{spec_text}\n"

    target_files = milestone.get("target_files", [])
    files_context = ""
    if target_files:
        files_context = (
            "\nTARGET FILES (focus on these patterns):\n"
            + "\n".join([f"- {pattern}" for pattern in target_files])
        )

    if current_files:
        files_context += f"\n\nCURRENT FILE CONTENTS (relevant snippets only):\n{current_files}"

    rules_text = "\n".join([f"- {rule}" for rule in repo_rules])

    return f"""You are a code modification agent for the Reclaim repository. Complete the milestone below.

REPO RULES (CRITICAL - MUST FOLLOW):
{rules_text}

MILESTONE:
Title: {milestone['title']}
Type: {milestone.get('type', 'feat')}
Acceptance commands (all must pass):
{acceptance}
{spec_block}{files_context}

CONSTRAINTS:
- Maximum {max_files} files changed
- Maximum {max_lines} lines net change (additions - deletions)

OUTPUT FORMAT (STRICT):
- Output ONLY a valid unified diff patch that `git apply` can parse.
- Use REAL line numbers in hunk headers: @@ -10,5 +10,8 @@ NOT @@ ... @@
- Include sufficient context lines (at least 3 before and after changes).
- NO explanations, NO markdown fences, NO commentary, NO placeholders.
- Start immediately with:  --- a/path/to/file.ext

EXAMPLE of correct format:
--- a/app/src/example.ts
+++ b/app/src/example.ts
@@ -15,7 +15,9 @@ export function example() {{
   const x = 1;
   const y = 2;
+  const z = 3;
   return x + y;
 }}

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


def _safe_json_preview(obj: Any, limit: int = 2000) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:limit]
    except Exception:
        return str(obj)[:limit]


def _extract_text_from_responses_api(payload: dict) -> str:
    """
    Extract the primary text output from OpenAI Responses API JSON.

    Common shapes:
    - payload["output_text"] (sometimes present)
    - payload["output"] = [
        { "type": "message", "content": [ { "type": "output_text", "text": "..." }, ... ] },
        { "type": "reasoning", ... }
      ]
    """
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()

    out = payload.get("output")
    if isinstance(out, list):
        parts: List[str] = []
        for item in out:
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


def _sanitize_to_unified_diff(text: str) -> str:
    """
    Ensure we return only the unified diff part starting at the first '--- a/' line.
    If not found, return the original (caller will treat empty/invalid appropriately).
    """
    if not text:
        return ""

    idx = text.find("--- a/")
    if idx == -1:
        # Some diffs may start with '---' then '+++', but we insist on '--- a/' for safety.
        idx = text.find("---\n+++")
        if idx == -1:
            return text.strip()
    return text[idx:].strip()


def _is_incomplete_max_tokens(payload: dict) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("status") == "incomplete"
        and isinstance(payload.get("incomplete_details"), dict)
        and payload["incomplete_details"].get("reason") == "max_output_tokens"
    )


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

    def _dbg(msg: str) -> None:
        if debug:
            print(msg)

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

    # Optional temperature: OFF by default, only enabled if you explicitly set OPENAI_ENABLE_TEMPERATURE=1
    enable_temp = _env_flag("OPENAI_ENABLE_TEMPERATURE")
    temp = _env_float("OPENAI_TEMPERATURE") if enable_temp else None

    # Optional verbosity: default OFF (because your codex runs showed it rejects low and only supports medium).
    # If you set OPENAI_TEXT_VERBOSITY, we will try it, but auto-remove on model rejection.
    verbosity = os.getenv("OPENAI_TEXT_VERBOSITY", "").strip() or None

    # Base request body (Responses API)
    data: Dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": "You output ONLY unified diff patches. Start immediately with '--- a/...'. No explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "text"}},
    }

    if verbosity:
        data["text"]["verbosity"] = verbosity  # may be rejected; we handle it below

    if temp is not None:
        data["temperature"] = temp  # may be rejected; we handle it below

    def _post(req_data: Dict[str, Any], timeout_pair: Tuple[int, int]) -> Tuple[int, Optional[dict], str]:
        resp = requests.post(url, headers=headers, json=req_data, timeout=timeout_pair)
        status = resp.status_code
        if status == 200:
            try:
                return status, resp.json(), ""
            except Exception:
                return status, None, resp.text
        else:
            try:
                return status, resp.json(), ""
            except Exception:
                return status, None, resp.text

    last_err: Optional[dict] = None
    last_raw_text: str = ""

    for attempt in range(1, retries + 1):
        try:
            _dbg(
                f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} "
                f"attempt={attempt}/{retries} max_output_tokens={data.get('max_output_tokens')} "
                f"timeout=({connect_timeout_s}, {read_timeout_s})"
            )

            status, payload, raw_text = _post(data, (connect_timeout_s, read_timeout_s))
            last_raw_text = raw_text

            if status == 200 and isinstance(payload, dict):
                if debug:
                    print(f"[AGENT_DEBUG] OpenAI raw response (first 2000 chars): {_safe_json_preview(payload, 2000)}")

                content = _extract_text_from_responses_api(payload)
                preview = (content[:400] if content else "").replace("\n", "\\n")
                _dbg(f"[AGENT_DEBUG] OpenAI content preview (first 400 chars): {preview}")

                # If we got something, sanitize to diff and return.
                if content and content.strip():
                    clean = _sanitize_to_unified_diff(content)
                    return clean.strip() if clean and clean.strip() else None

                # If we got no text but the response is incomplete due to token cap,
                # bump tokens and (optionally) do a continuation with previous_response_id.
                if _is_incomplete_max_tokens(payload):
                    current = int(data.get("max_output_tokens") or max_output_tokens)
                    bumped = _clamp(max(current * 3, current + 2000), 512, cap_tokens)
                    if bumped > current:
                        _dbg(f"[AGENT_DEBUG] Incomplete due to max_output_tokens. Bumping {current} -> {bumped} and retrying.")
                        data["max_output_tokens"] = bumped
                        # increase read timeout as well
                        read_timeout_s = max(read_timeout_s, _env_int("OPENAI_READ_TIMEOUT_S", 120))
                        # try again immediately
                        continue

                    # If already at cap and still no text, try continuation once.
                    resp_id = payload.get("id")
                    if isinstance(resp_id, str) and resp_id:
                        _dbg("[AGENT_DEBUG] Still no text at token cap; attempting continuation via previous_response_id.")
                        cont_data = dict(data)
                        cont_data["previous_response_id"] = resp_id
                        cont_data["input"] = [
                            {
                                "role": "user",
                                "content": "Continue. Output ONLY the unified diff patch starting with '--- a/...'.",
                            }
                        ]
                        # more time for continuation
                        status2, payload2, raw2 = _post(cont_data, (connect_timeout_s, max(read_timeout_s, 300)))
                        if status2 == 200 and isinstance(payload2, dict):
                            if debug:
                                print(f"[AGENT_DEBUG] OpenAI continuation raw (first 2000): {_safe_json_preview(payload2, 2000)}")
                            cont_text = _extract_text_from_responses_api(payload2)
                            cont_preview = (cont_text[:400] if cont_text else "").replace("\n", "\\n")
                            _dbg(f"[AGENT_DEBUG] Continuation content preview (first 400 chars): {cont_preview}")
                            if cont_text and cont_text.strip():
                                clean2 = _sanitize_to_unified_diff(cont_text)
                                return clean2.strip() if clean2 and clean2.strip() else None

                # No usable content
                return None

            # Non-200: handle errors and retry where sensible
            last_err = payload if isinstance(payload, dict) else None
            _dbg(f"[AGENT_DEBUG] OpenAI non-200 status={status}")

            if debug:
                if isinstance(last_err, dict):
                    _dbg(f"[AGENT_DEBUG] OpenAI error JSON: {last_err}")
                elif last_raw_text:
                    _dbg(f"[AGENT_DEBUG] OpenAI error text: {last_raw_text[:2000]}")

            # Parse error fields
            err = (last_err or {}).get("error") if isinstance(last_err, dict) else None
            err_type = err.get("type") if isinstance(err, dict) else None
            err_code = err.get("code") if isinstance(err, dict) else None
            err_param = err.get("param") if isinstance(err, dict) else None
            err_msg = (err.get("message") if isinstance(err, dict) else "") or ""

            # Quota: do not retry
            if err_code == "insufficient_quota" or err_type == "insufficient_quota":
                print("OpenAI quota exhausted / billing not active for this API key (insufficient_quota).")
                return None

            # Auto-remove unsupported temperature
            if (err_param == "temperature" or "temperature" in err_msg.lower()) and "temperature" in data:
                _dbg("[AGENT_DEBUG] Model rejected temperature; removing temperature and retrying.")
                data.pop("temperature", None)
                if attempt < retries:
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 30)
                    continue

            # Auto-fix verbosity problems: remove verbosity key entirely if rejected
            if (err_param == "text.verbosity" or "verbosity" in err_msg.lower()) and isinstance(data.get("text"), dict):
                if "verbosity" in data["text"]:
                    _dbg("[AGENT_DEBUG] Model rejected text.verbosity; removing verbosity and retrying.")
                    data["text"].pop("verbosity", None)
                    if attempt < retries:
                        time.sleep(backoff_s)
                        backoff_s = min(backoff_s * 2, 30)
                        continue

            # Retry on transient status codes
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