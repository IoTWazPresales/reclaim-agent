"""Main orchestration logic for the agent."""

from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import shlex

from .config import Config
from .github_api import GitHubAPI
from .prompts import build_fix_prompt, build_milestone_prompt, call_openai
from .milestones import get_next_todo_milestone, update_milestone_status


class Runner:
    """Main runner for agent operations."""

    def __init__(self, config: Config):
        self.config = config
        self.github = GitHubAPI(config.github_token, config.repo_name)
        self.repo_path = Path(config.repo_path) if config.repo_path else None

    # -------------------------
    # Debug / strict helpers
    # -------------------------
    def _env_flag(self, name: str) -> bool:
        return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")

    def _debug_enabled(self) -> bool:
        return self._env_flag("AGENT_DEBUG")

    def _strict_enabled(self) -> bool:
        return self._env_flag("AGENT_STRICT")

    def _fail(self, msg: str) -> None:
        """Centralized failure behavior."""
        print(msg)
        if self._strict_enabled():
            raise RuntimeError(msg)

    # -------------------------
    # Command runner
    # -------------------------
    def _looks_like_shell(self, cmd: str) -> bool:
        """Detect common shell syntax that requires shell=True (bash)."""
        s = cmd.strip()
        if not s:
            return False
        shell_tokens = ("&&", "||", ";", "|", ">", "<", "$(", "`")
        if any(tok in s for tok in shell_tokens):
            return True
        # "cd something" only makes sense in a shell
        if s.startswith("cd "):
            return True
        return False

    def _run_cmd(
        self,
        cmd: str | List[str],
        cwd: Optional[Path] = None,
        timeout: int = 300,
        label: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Run a command with robust handling for:
        - list args (shell=False)
        - string commands with shell operators like `&&` (shell=True with bash)
        """
        workdir = str(cwd) if cwd else None
        debug = self._debug_enabled()

        if isinstance(cmd, list):
            args = cmd
            shell_mode = False
            display = " ".join(args)
        else:
            shell_mode = self._looks_like_shell(cmd)
            if shell_mode:
                args = cmd  # string
                display = cmd
            else:
                args = shlex.split(cmd)
                display = cmd

        if debug:
            tag = f"[{label}]" if label else ""
            print(f"[AGENT_DEBUG] Running: {display} (cwd={workdir}) {tag}")

        result = subprocess.run(
            args,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell_mode,
            executable="/usr/bin/bash" if shell_mode else None,
        )

        if debug:
            out = (result.stdout or "")[:1200]
            err = (result.stderr or "")[:1200]
            if out:
                print("[AGENT_DEBUG] STDOUT (first 1200 chars):")
                print(out)
            if err:
                print("[AGENT_DEBUG] STDERR (first 1200 chars):")
                print(err)

        return result

    # -------------------------
    # Truth checks / patch apply
    # -------------------------
    def run_truth_checks(self) -> List[Dict[str, Any]]:
        """Run truth checks and return failing checks."""
        if not self.repo_path:
            return []

        failing: List[Dict[str, Any]] = []
        app_path = self.repo_path / "app"
        if not app_path.exists():
            return []

        for check in self.config.truth_checks:
            try:
                result = self._run_cmd(
                    check["command"],
                    cwd=app_path,
                    timeout=300,
                    label=check.get("name", "truth-check"),
                )
                if result.returncode != 0:
                    failing.append(
                        {
                            "name": check["name"],
                            "command": check["command"],
                            "error": (result.stderr or "")[:500],
                            "output": (result.stdout or "")[:500],
                        }
                    )
            except subprocess.TimeoutExpired:
                failing.append(
                    {
                        "name": check["name"],
                        "command": check["command"],
                        "error": "Command timed out",
                        "output": "",
                    }
                )
            except Exception as e:
                failing.append(
                    {
                        "name": check["name"],
                        "command": check["command"],
                        "error": str(e),
                        "output": "",
                    }
                )

        return failing

    def apply_patch(self, patch: str, base_path: Path) -> Tuple[bool, Optional[str]]:
        """Apply unified diff patch to repository."""
        lines = patch.strip().split("\n")
        diff_start = None
        for i, line in enumerate(lines):
            if line.startswith("--- ") or line.startswith("+++ "):
                diff_start = i
                break

        if diff_start is None:
            return False, "No diff found in patch"

        clean_patch = "\n".join(lines[diff_start:])

        # ---------------------------
        # Basic structural validation
        # ---------------------------
        import re

        hunk_header_pattern = re.compile(r"^@@\s+.*\s+@@", re.MULTILINE)
        placeholder_hunk_pattern = re.compile(r"^@@\s+\.\.\..*@@", re.MULTILINE)
        zero_length_hunk_pattern = re.compile(r"^@@\s+-\d+,0\s+\+\d+,0\s+@@", re.MULTILINE)

        # Reject obvious placeholder hunks (e.g. @@ ... @@)
        if placeholder_hunk_pattern.search(clean_patch):
            return False, (
                "Patch contains placeholder hunk headers (@@ ... @@). "
                "LLM must provide REAL line numbers like @@ -10,5 +10,8 @@."
            )

        # Reject zero-length hunks like @@ -1,0 +0,0 @@ (no actual changes)
        if zero_length_hunk_pattern.search(clean_patch):
            return False, (
                "Patch contains zero-length hunks (e.g. @@ -1,0 +0,0 @@). "
                "Each hunk must add and/or remove at least one line."
            )

        # Check if patch has any hunk headers at all
        if not hunk_header_pattern.search(clean_patch):
            return False, "Patch is missing hunk headers. Must include line numbers like @@ -10,5 +10,8 @@."

        # ---------------------------
        # File header sanity checks
        # ---------------------------
        first_header = None
        for l in clean_patch.split("\n"):
            if l.startswith("--- "):
                first_header = l
                break

        if first_header:
            header_path = first_header[4:].strip()  # strip leading '--- '
            # Normalize typical git diff prefix
            if header_path.startswith("a/"):
                header_path = header_path[2:]

            # Reject obvious placeholder filenames
            lowered = header_path.lower()
            if "placeholder" in lowered or "dummy" in lowered or "example" == lowered:
                return False, (
                    f"Patch targets placeholder file path '{header_path}'. "
                    "LLM must use REAL file paths from the repository (see TARGET FILES and CURRENT FILE CONTENTS)."
                )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(clean_patch)
            patch_path = f.name

        try:
            check = self._run_cmd(
                ["git", "apply", "--check", patch_path],
                cwd=base_path,
                timeout=120,
                label="git apply --check",
            )
            if check.returncode != 0:
                return False, f"Patch check failed: {check.stderr}"

            apply = self._run_cmd(
                ["git", "apply", patch_path],
                cwd=base_path,
                timeout=120,
                label="git apply",
            )
            if apply.returncode != 0:
                return False, f"Patch apply failed: {apply.stderr}"

            return True, None
        finally:
            try:
                os.unlink(patch_path)
            except Exception:
                pass

    # -------------------------
    # Modes
    # -------------------------
    def _branch_exists(self, branch_name: str) -> bool:
        """Check if a local git branch already exists."""
        if not self.repo_path:
            return False
        result = self._run_cmd(
            ["git", "rev-parse", "--verify", branch_name],
            cwd=self.repo_path,
            timeout=30,
            label="git rev-parse branch-exists",
        )
        return result.returncode == 0

    def _ensure_branch_checked_out(self, branch_name: str, base_branch: str) -> bool:
        """
        Ensure a working branch exists and is checked out.

        - Reuse existing branch if present.
        - Otherwise create it from base_branch and check it out.
        """
        if not self.repo_path:
            return False

        if self._branch_exists(branch_name):
            # Reuse existing branch locally
            result = self._run_cmd(
                ["git", "checkout", branch_name],
                cwd=self.repo_path,
                timeout=60,
                label="git checkout existing",
            )
            return result.returncode == 0

        # Create branch in remote (idempotent: create_branch tolerates 422)
        self.github.create_branch(branch_name, base_branch)
        result = self._run_cmd(
            ["git", "checkout", "-b", branch_name],
            cwd=self.repo_path,
            timeout=60,
            label="git checkout -b",
        )
        return result.returncode == 0

    def run_fix_mode(self) -> Optional[str]:
        """Run in fix mode - fix failing truth checks."""
        print("Running truth checks...")
        failing = self.run_truth_checks()

        if not failing:
            print("All truth checks passing - no fix needed")
            return None

        print(f"Found {len(failing)} failing checks")

        prompt = build_fix_prompt(
            failing,
            self.config.repo_rules,
            self.config.max_files,
            self.config.max_lines,
        )

        print("Calling OpenAI API for fix patch...")
        try:
            patch = call_openai(prompt, self.config.openai_api_key)
        except Exception as e:
            print("Failed to generate patch (exception thrown during fix mode)")
            if self._debug_enabled():
                print("=== OpenAI exception (fix mode) ===")
                traceback.print_exc()
            self._fail(f"OpenAI exception in fix mode: {type(e).__name__}: {e}")
            return None

        if not patch:
            print("Failed to generate patch (empty response in fix mode)")
            self._fail("OpenAI returned empty patch in fix mode")
            return None

        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-fix-truth-checks"

        # Idempotency: if a PR already exists for this branch, do not create a new one.
        existing_pr = self.github.get_pr_by_branch(branch_name)
        if existing_pr:
            print(f"Existing PR for fix branch found, skipping new run: {existing_pr['html_url']}")
            return existing_pr.get("html_url")

        if not self._ensure_branch_checked_out(branch_name, self.config.default_branch):
            self._fail(f"Failed to prepare branch {branch_name} for fix mode")
            return None

        success, error = self.apply_patch(patch, self.repo_path)
        if not success:
            self._fail(f"Patch apply failed in fix mode: {error}")
            return None

        print("Verifying fixes...")
        still_failing = self.run_truth_checks()
        if still_failing:
            print("Truth checks still failing after fix patch:")
            for check in still_failing:
                print(f"- {check.get('name')}: {check.get('error')}")
            self._fail("Truth checks still failing after fix patch")
            return None

        self._run_cmd(["git", "add", "-A"], cwd=self.repo_path, timeout=60, label="git add")
        self._run_cmd(["git", "commit", "-m", "fix: resolve failing truth checks"], cwd=self.repo_path, timeout=60, label="git commit")
        self._run_cmd(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, timeout=120, label="git push")

        check_outputs = "\n".join([f"- {check['name']}: ✅ PASS" for check in self.config.truth_checks])

        pr_body = f"""## Summary
Fixed failing truth checks in the repository.

## Root Cause
The following truth checks were failing:
{chr(10).join([f"- {check['name']}: {check.get('error', 'Failed')}" for check in failing])}

## Changes
- Applied LLM-generated patch to resolve failures
- All truth checks now passing

## Verification
{check_outputs}

## Files Changed
See diff for details.
"""

        pr = self.github.create_pr(
            title="fix: resolve failing truth checks",
            body=pr_body,
            head=branch_name,
            base=self.config.default_branch,
        )

        if pr:
            print(f"Created PR: {pr['html_url']}")
            return pr["html_url"]

        return None

    def run_milestone_mode(self) -> Optional[str]:
        """Run in milestone mode - complete next todo milestone."""
        milestone = get_next_todo_milestone(self.config.milestones)
        if not milestone:
            print("No todo milestones found")
            return None

        milestone_id = milestone["id"]
        print(f"Processing milestone: {milestone['title']} ({milestone_id})")

        # Track attempts and stop after configured max_attempts to avoid infinite retries.
        attempts = int(milestone.get("attempts", 0)) + 1
        milestone["attempts"] = attempts
        if attempts > self.config.max_attempts:
            update_milestone_status(
                self.config.milestones,
                milestone_id,
                "blocked",
                f"Exceeded max_attempts ({self.config.max_attempts})",
            )
            self.config.save()
            print(f"Milestone {milestone_id} blocked due to exceeding max attempts")
            return None

        update_milestone_status(self.config.milestones, milestone_id, "in_progress")
        self.config.save()

        # Optionally gather a small amount of file context for the LLM based on target_files.
        current_files_snippet: Optional[str] = None
        try:
            if self.repo_path and milestone.get("target_files"):
                snippets: List[str] = []
                max_files = 5
                max_chars_per_file = 2000
                matched = 0
                for pattern in milestone["target_files"]:
                    # Use git ls-files for globbing inside the repo to respect .gitignore.
                    result = self._run_cmd(
                        ["git", "ls-files", pattern],
                        cwd=self.repo_path,
                        timeout=60,
                        label=f"git ls-files {pattern}",
                    )
                    if result.returncode != 0:
                        continue
                    for rel_path in (result.stdout or "").splitlines():
                        if not rel_path.strip():
                            continue
                        file_path = (self.repo_path / rel_path.strip())
                        if not file_path.is_file():
                            continue
                        try:
                            text = file_path.read_text(encoding="utf-8")
                        except Exception:
                            continue
                        snippet = text[:max_chars_per_file]
                        snippets.append(f"--- FILE: {rel_path.strip()} ---\n{snippet}\n")
                        matched += 1
                        if matched >= max_files:
                            break
                    if matched >= max_files:
                        break
                if snippets:
                    current_files_snippet = "\n".join(snippets)
        except Exception:
            # Context gathering should never break the run.
            current_files_snippet = None

        prompt = build_milestone_prompt(
            milestone,
            self.config.repo_rules,
            self.config.max_files,
            self.config.max_lines,
            current_files=current_files_snippet,
        )

        print("Calling OpenAI API for milestone patch...")
        try:
            patch = call_openai(prompt, self.config.openai_api_key)
        except Exception as e:
            print("Failed to generate patch (exception thrown in milestone mode)")
            if self._debug_enabled():
                print("=== OpenAI exception (milestone mode) ===")
                traceback.print_exc()

            update_milestone_status(self.config.milestones, milestone_id, "blocked", f"OpenAI exception: {type(e).__name__}: {e}")
            self.config.save()
            self._fail(f"OpenAI exception in milestone mode: {type(e).__name__}: {e}")
            return None

        if not patch:
            print("Failed to generate patch (empty response in milestone mode)")
            update_milestone_status(self.config.milestones, milestone_id, "blocked", "Failed to generate patch: empty response from call_openai")
            self.config.save()
            self._fail("OpenAI returned empty patch in milestone mode")
            return None

        branch_slug = milestone_id.replace("_", "-")
        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-{branch_slug}"

        # Idempotency: if a PR already exists for this milestone branch, do not recreate it.
        existing_pr = self.github.get_pr_by_branch(branch_name)
        if existing_pr:
            print(f"Existing PR for milestone found, marking as done: {existing_pr['html_url']}")
            update_milestone_status(self.config.milestones, milestone_id, "done")
            self.config.save()
            return existing_pr.get("html_url")

        if not self._ensure_branch_checked_out(branch_name, self.config.default_branch):
            update_milestone_status(self.config.milestones, milestone_id, "blocked", f"Failed to prepare branch {branch_name}")
            self.config.save()
            self._fail(f"Failed to prepare branch {branch_name} in milestone mode")
            return None

        success, error = self.apply_patch(patch, self.repo_path)
        if not success:
            update_milestone_status(self.config.milestones, milestone_id, "blocked", f"Patch apply failed: {error}")
            self.config.save()
            self._fail(f"Patch apply failed in milestone mode: {error}")
            return None

        print("Verifying acceptance criteria...")
        # Acceptance commands are assumed relative to repo root; they can do their own `cd`.
        app_path = self.repo_path
        for cmd in milestone.get("acceptance", []):
            result = self._run_cmd(cmd, cwd=app_path, timeout=300, label=f"acceptance: {cmd}")
            if result.returncode != 0:
                reason = f"Acceptance failed: {cmd}\nSTDOUT:\n{(result.stdout or '')[:500]}\nSTDERR:\n{(result.stderr or '')[:500]}"
                update_milestone_status(self.config.milestones, milestone_id, "blocked", reason)
                self.config.save()
                self._fail("Acceptance criteria not met in milestone mode")
                return None

        self._run_cmd(["git", "add", "-A"], cwd=self.repo_path, timeout=60, label="git add")
        self._run_cmd(["git", "commit", "-m", f"feat: {milestone['title']}"], cwd=self.repo_path, timeout=60, label="git commit")
        self._run_cmd(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, timeout=120, label="git push")

        acceptance_outputs = "\n".join([f"- `{cmd}`: ✅ PASS" for cmd in milestone.get("acceptance", [])])

        pr_body = f"""## Summary
Completed milestone: {milestone['title']}

## Root Cause
Milestone task: {milestone.get('type', 'feature')} - {milestone['title']}

## Changes
- Applied LLM-generated patch to complete milestone
- All acceptance criteria now passing

## Verification
{acceptance_outputs}

## Files Changed
See diff for details.
"""

        pr = self.github.create_pr(
            title=f"feat: {milestone['title']}",
            body=pr_body,
            head=branch_name,
            base=self.config.default_branch,
        )

        if pr:
            print(f"Created PR: {pr['html_url']}")
            update_milestone_status(self.config.milestones, milestone_id, "done")
            self.config.save()
            return pr["html_url"]

        return None

    def run(self, mode: str = "auto") -> Optional[str]:
        """Run agent in specified mode."""
        if not self.repo_path or not self.repo_path.exists():
            print("Repository path not found")
            return None

        # Ensure we're on default branch
        self._run_cmd(["git", "checkout", self.config.default_branch], cwd=self.repo_path, timeout=60, label="git checkout default")
        self._run_cmd(["git", "pull"], cwd=self.repo_path, timeout=120, label="git pull")

        if mode in ("fix", "auto"):
            failing = self.run_truth_checks()
            if failing:
                return self.run_fix_mode()

        if mode in ("milestone", "auto"):
            milestone = get_next_todo_milestone(self.config.milestones)
            if milestone:
                result = self.run_milestone_mode()
                if result and milestone.get("stop_feature"):
                    print("Stop feature enabled - stopping after milestone")
                    return result
                return result

        print("No work needed - repo is green and no milestones")
        return None
