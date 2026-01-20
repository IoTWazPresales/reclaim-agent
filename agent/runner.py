"""Main orchestration logic for the agent."""

from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

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

    def _dbg(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg)

    def _fail(self, msg: str) -> None:
        """
        Centralized failure behavior.
        - In strict mode: raise (fails CI with non-zero exit)
        - Otherwise: just print and return
        """
        print(msg)
        if self._strict_enabled():
            raise RuntimeError(msg)

    def _run_cmd(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: int = 300,
        label: str = "",
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Wrapper around subprocess.run with consistent debug output.
        """
        self._dbg(f"[AGENT_DEBUG] Running: {' '.join(cmd)} (cwd={cwd}) {f'[{label}]' if label else ''}")
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=capture,
            text=True,
            timeout=timeout,
        )
        if self._debug_enabled() and capture:
            out = (result.stdout or "")[:1200]
            err = (result.stderr or "")[:1200]
            if out:
                self._dbg("[AGENT_DEBUG] STDOUT (first 1200 chars):\n" + out)
            if err:
                self._dbg("[AGENT_DEBUG] STDERR (first 1200 chars):\n" + err)
        return result

    # -------------------------
    # Truth checks
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
            cmd = check["command"].split()
            try:
                result = self._run_cmd(cmd, cwd=app_path, timeout=300, label=check["name"])
                if result.returncode != 0:
                    failing.append({
                        "name": check["name"],
                        "command": check["command"],
                        "error": (result.stderr or "")[:1000],
                        "output": (result.stdout or "")[:1000],
                    })
            except subprocess.TimeoutExpired:
                failing.append({
                    "name": check["name"],
                    "command": check["command"],
                    "error": "Command timed out",
                    "output": "",
                })
            except Exception as e:
                failing.append({
                    "name": check["name"],
                    "command": check["command"],
                    "error": str(e),
                    "output": "",
                })

        return failing

    # -------------------------
    # Patch application
    # -------------------------

    def apply_patch(self, patch: str, base_path: Path) -> Tuple[bool, Optional[str]]:
        """Apply unified diff patch to repository."""
        if not patch or not patch.strip():
            return False, "Empty patch"

        # Extract diff start
        lines = patch.strip("\n").split("\n")
        diff_start = None
        for i, line in enumerate(lines):
            if line.startswith("--- a/"):
                diff_start = i
                break

        if diff_start is None:
            return False, "No diff found in patch (missing '--- a/...')"

        clean_patch = "\n".join(lines[diff_start:]).rstrip() + "\n"  # ensure newline at end

        # Write patch to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(clean_patch)
            patch_path = f.name

        try:
            # Check patch
            result = self._run_cmd(["git", "apply", "--check", patch_path], cwd=base_path, timeout=120, label="git apply --check")
            if result.returncode != 0:
                return False, f"Patch check failed: {result.stderr or result.stdout}"

            # Apply patch
            result = self._run_cmd(["git", "apply", patch_path], cwd=base_path, timeout=120, label="git apply")
            if result.returncode != 0:
                return False, f"Patch apply failed: {result.stderr or result.stdout}"

            return True, None
        finally:
            try:
                os.unlink(patch_path)
            except Exception:
                pass

    def _looks_like_corrupt_patch_error(self, err: str) -> bool:
        if not err:
            return False
        e = err.lower()
        return (
            "corrupt patch" in e
            or "patch unexpectedly ends" in e
            or "malformed patch" in e
            or "error: corrupt patch" in e
        )

    def _regen_prompt_for_corrupt_patch(self, original_prompt: str, error: str) -> str:
        """
        Ask model to re-emit a complete patch.
        Keep it short and strict. We provide the failure hint and hard requirements.
        """
        return (
            original_prompt
            + "\n\n"
            + "IMPORTANT:\n"
            + "The previous unified diff was TRUNCATED or CORRUPT and could not be applied by `git apply`.\n"
            + f"git error was: {error[:300]}\n"
            + "Re-output the COMPLETE unified diff patch again.\n"
            + "Rules:\n"
            + "- Output ONLY the unified diff.\n"
            + "- Ensure ALL hunks are complete (no cut-off lines).\n"
            + "- Ensure the patch ends cleanly with a newline.\n"
            + "- Start with `--- a/...`.\n"
        )

    # -------------------------
    # Modes
    # -------------------------

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

        # Create branch
        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-fix-truth-checks"
        self.github.create_branch(branch_name, self.config.default_branch)

        # Checkout branch in repo
        self._run_cmd(["git", "checkout", "-b", branch_name], cwd=self.repo_path, timeout=120, label="git checkout -b")

        # Apply patch (try once more if corrupt)
        success, error = self.apply_patch(patch, self.repo_path)
        if not success and error and self._looks_like_corrupt_patch_error(error):
            print("Patch looked corrupt/truncated. Regenerating once...")
            regen_prompt = self._regen_prompt_for_corrupt_patch(prompt, error)
            patch2 = call_openai(regen_prompt, self.config.openai_api_key)
            if patch2:
                success, error = self.apply_patch(patch2, self.repo_path)

        if not success:
            print(f"Failed to apply patch: {error}")
            self._fail(f"Patch apply failed in fix mode: {error}")
            return None

        # Verify truth checks pass
        print("Verifying fixes...")
        still_failing = self.run_truth_checks()
        if still_failing:
            print("Truth checks still failing after patch")
            self._fail("Truth checks still failing after fix patch")
            return None

        # Commit changes
        self._run_cmd(["git", "add", "-A"], cwd=self.repo_path, timeout=120, label="git add")
        self._run_cmd(["git", "commit", "-m", "fix: resolve failing truth checks"], cwd=self.repo_path, timeout=120, label="git commit")
        self._run_cmd(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, timeout=180, label="git push")

        # Create PR
        pr_body = (
            "## Summary\n"
            "Fixed failing truth checks in the repository.\n\n"
            "## Verification\n"
            + "\n".join([f"- {c['name']}: ✅ PASS" for c in self.config.truth_checks])
            + "\n"
        )

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

        # Mark in progress
        update_milestone_status(self.config.milestones, milestone_id, "in_progress")
        self.config.save()

        prompt = build_milestone_prompt(
            milestone,
            self.config.repo_rules,
            self.config.max_files,
            self.config.max_lines,
        )

        print("Calling OpenAI API for milestone patch...")
        try:
            patch = call_openai(prompt, self.config.openai_api_key)
        except Exception as e:
            msg = f"OpenAI exception in milestone mode: {type(e).__name__}: {e}"
            print("Failed to generate patch (exception thrown in milestone mode)")
            if self._debug_enabled():
                print("=== OpenAI exception (milestone mode) ===")
                traceback.print_exc()
            update_milestone_status(self.config.milestones, milestone_id, "blocked", msg)
            self.config.save()
            self._fail(msg)
            return None

        if not patch:
            msg = "Failed to generate patch: empty response from call_openai"
            print("Failed to generate patch (empty response in milestone mode)")
            update_milestone_status(self.config.milestones, milestone_id, "blocked", msg)
            self.config.save()
            self._fail("OpenAI returned empty patch in milestone mode")
            return None

        # Create branch
        branch_slug = milestone_id.replace("_", "-")
        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-{branch_slug}"
        self.github.create_branch(branch_name, self.config.default_branch)

        # Checkout branch
        self._run_cmd(["git", "checkout", "-b", branch_name], cwd=self.repo_path, timeout=120, label="git checkout -b")

        # Apply patch (try once more if corrupt)
        success, error = self.apply_patch(patch, self.repo_path)
        if not success and error and self._looks_like_corrupt_patch_error(error):
            print("Patch looked corrupt/truncated. Regenerating once...")
            regen_prompt = self._regen_prompt_for_corrupt_patch(prompt, error)
            patch2 = call_openai(regen_prompt, self.config.openai_api_key)
            if patch2:
                success, error = self.apply_patch(patch2, self.repo_path)

        if not success:
            msg = f"Patch apply failed in milestone mode: {error}"
            print(msg)
            update_milestone_status(self.config.milestones, milestone_id, "blocked", msg)
            self.config.save()
            self._fail(msg)
            return None

        # Verify acceptance criteria
        print("Verifying acceptance criteria...")
        app_path = self.repo_path / "app"

        for cmd in milestone.get("acceptance", []):
            cmd_parts = cmd.split()
            result = self._run_cmd(cmd_parts, cwd=app_path, timeout=300, label=f"acceptance: {cmd}")
            if result.returncode != 0:
                msg = f"Acceptance criteria not met (failed: {cmd})"
                print(msg)
                update_milestone_status(self.config.milestones, milestone_id, "blocked", msg)
                self.config.save()
                self._fail(msg)
                return None

        # Commit changes
        self._run_cmd(["git", "add", "-A"], cwd=self.repo_path, timeout=120, label="git add")
        self._run_cmd(["git", "commit", "-m", f"feat: {milestone['title']}"], cwd=self.repo_path, timeout=120, label="git commit")
        self._run_cmd(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, timeout=180, label="git push")

        # Create PR
        acceptance_outputs = "\n".join([f"- `{cmd}`: ✅ PASS" for cmd in milestone.get("acceptance", [])])
        pr_body = f"""## Summary
Completed milestone: {milestone['title']}

## Verification
{acceptance_outputs}
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

        msg = "Failed to create PR (GitHub API returned no PR object)"
        update_milestone_status(self.config.milestones, milestone_id, "blocked", msg)
        self.config.save()
        self._fail(msg)
        return None

    # -------------------------
    # Main entry
    # -------------------------

    def run(self, mode: str = "auto") -> Optional[str]:
        """Run agent in specified mode."""
        if not self.repo_path or not self.repo_path.exists():
            print("Repository path not found")
            return None

        # Ensure we're on default branch and updated
        res = self._run_cmd(["git", "checkout", self.config.default_branch], cwd=self.repo_path, timeout=120, label="git checkout default")
        if res.returncode != 0:
            self._fail(f"Failed to checkout default branch: {res.stderr or res.stdout}")
            return None

        res = self._run_cmd(["git", "pull"], cwd=self.repo_path, timeout=180, label="git pull")
        if res.returncode != 0:
            self._fail(f"Failed to pull default branch: {res.stderr or res.stdout}")
            return None

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