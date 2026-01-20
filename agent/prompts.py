"""Prompt templates for LLM interactions.

Defensive OpenAI caller:
- Uses Responses API (v1/responses).
- Extracts text from multiple payload shapes (including payload["text"]).
- Retries automatically when the response is incomplete due to max_output_tokens.
- Uses adaptive timeouts for large max_output_tokens to avoid ReadTimeout.
- Nudges model to emit TEXT (diff) rather than spending all tokens on reasoning.
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

    # Strong “emit diff now” directive (helps prevent wasting 100% on reasoning)
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
    """
    Extract primary text from Responses API payload.

    Supports:
    - payload["output_text"] (convenience)
    - payload["text"]["value"] / payload["text"]["content"]
    - payload["output"][...]["content"][...]["text"]
    - payload["output"][...]["text"]
    """
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
        chunks: List[str] = []
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


def call_openai(prompt: str, api_key: str) -> Optional[str]:
    """
    Call OpenAI Responses API and return unified diff text.

    Reliability:
    - Adaptive timeouts (bigger outputs take longer).
    - If response is incomplete due to max_output_tokens, bump gradually (4k->8k->16k by default).
    - Nudges model toward text output: reasoning effort low, verbosity low.
    """
    import requests

    debug = _env_flag("AGENT_DEBUG")

    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4.1"
    base = os.getenv("OPENAI_ENDPOINT", "").strip() or "https://api.openai.com"
    url = f"{base}/v1/responses"

    # timeouts: separate connect + read timeout for better control
    connect_timeout_s = 15
    read_timeout_s = 120  # default read timeout (bigger than 90)
    ct = os.getenv("OPENAI_CONNECT_TIMEOUT_S", "").strip()
    rt = os.getenv("OPENAI_READ_TIMEOUT_S", "").strip()
    if ct:
        try:
            connect_timeout_s = int(ct)
        except Exception:
            pass
    if rt:
        try:
            read_timeout_s = int(rt)
        except Exception:
            pass

    # output token budget
    max_out = 4000
    mo = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
    if mo:
        try:
            max_out = int(mo)
        except Exception:
            pass

    # cap and bump schedule
    bump_cap = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS_CAP", "").strip() or "20000")
    bump_steps_env = os.getenv("OPENAI_BUMP_STEPS", "").strip()
    if bump_steps_env:
        # e.g. "8000,12000,16000"
        try:
            bump_steps = [int(x.strip()) for x in bump_steps_env.split(",") if x.strip()]
            bump_steps = [x for x in bump_steps if x > 0]
        except Exception:
            bump_steps = [8000, 16000]
    else:
        bump_steps = [8000, 16000]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def make_data(max_output_tokens: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": "Output ONLY a unified diff patch. Start immediately with `--- a/`."},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,

            # These strongly bias the model away from spending everything on reasoning.
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
        }

        # Only include temperature if explicitly set; some models reject it.
        temp = os.getenv("OPENAI_TEMPERATURE", "").strip()
        if temp:
            try:
                data["temperature"] = float(temp)
            except ValueError:
                pass

        return data

    retries = 3
    backoff_s = 2

    # track which bump step we’re on
    bump_index = -1  # means "no bump applied yet"

    for attempt in range(1, retries + 1):
        try:
            # Adaptive read timeout: scale with max_out (very rough but works)
            # Example: 4k -> 120s, 8k -> 180s, 16k -> 300s
            adaptive_read = read_timeout_s
            if max_out >= 8000:
                adaptive_read = max(read_timeout_s, 180)
            if max_out >= 16000:
                adaptive_read = max(read_timeout_s, 300)

            timeout = (connect_timeout_s, adaptive_read)

            data = make_data(max_out)

            if debug:
                print(
                    f"[AGENT_DEBUG] OpenAI request: endpoint=v1/responses model={model} "
                    f"attempt={attempt}/{retries} max_output_tokens={max_out} timeout={timeout}"
                )

            response = requests.post(url, headers=headers, json=data, timeout=timeout)
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

                if extracted and extracted.strip():
                    return extracted.strip()

                # Handle incomplete due to max_output_tokens: bump gradually
                resp_status = payload.get("status")
                inc = payload.get("incomplete_details") or {}
                reason = inc.get("reason")

                if resp_status == "incomplete" and reason == "max_output_tokens":
                    # bump step
                    next_budget = None

                    # first try: pick next from bump_steps if available
                    if bump_index + 1 < len(bump_steps):
                        bump_index += 1
                        next_budget = bump_steps[bump_index]
                    else:
                        # fallback: x2, capped
                        next_budget = min(max_out * 2, bump_cap)

                    next_budget = min(next_budget, bump_cap)

                    if next_budget > max_out:
                        if debug:
                            print(f"[AGENT_DEBUG] Incomplete due to max_output_tokens. Bumping {max_out} -> {next_budget} and retrying.")
                        max_out = next_budget
                        # Retry immediately without consuming attempt count further (we just continue)
                        continue

                return None

            # Non-200 handling
            err_json = None
            if debug:
                print(f"[AGENT_DEBUG] OpenAI non-200 status={status_code}")
                try:
                    err_json = response.json()
                    print(f"[AGENT_DEBUG] OpenAI error JSON: {err_json}")
                except Exception:
                    print(f"[AGENT_DEBUG] OpenAI error text: {_safe_trunc(response.text, 2000)}")

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

            if status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue

            return None

        except requests.exceptions.ReadTimeout as e:
            if debug:
                print(f"[AGENT_DEBUG] OpenAI ReadTimeout: {e}")

            # On ReadTimeout, we retry, but also increase read timeout a bit for next attempt.
            # (Don’t mutate env; just local default ramp.)
            read_timeout_s = max(read_timeout_s, 240)

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