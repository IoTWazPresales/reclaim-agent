"""Main orchestration logic for the agent."""

import os
import subprocess
import tempfile
import traceback
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

    def _print_debug(self, msg: str) -> None:
        if self._debug_enabled():
            print(msg)

    def _run_cmd(
        self,
        cmd: List[str],
        cwd: Optional[Path],
        timeout: int = 300,
        check: bool = False,
        label: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a subprocess command with consistent capture + debug output.
        If check=True, raises RuntimeError on non-zero exit.
        """
        self._print_debug(f"[AGENT_DEBUG] Running: {' '.join(cmd)} (cwd={cwd})" + (f" [{label}]" if label else ""))

        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if self._debug_enabled():
            if result.stdout:
                print("[AGENT_DEBUG] STDOUT (first 1200 chars):")
                print(result.stdout[:1200])
            if result.stderr:
                print("[AGENT_DEBUG] STDERR (first 1200 chars):")
                print(result.stderr[:1200])

        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({label or 'subprocess'}): {' '.join(cmd)}\n"
                f"returncode={result.returncode}\n"
                f"stdout={(result.stdout or '')[:1200]}\n"
                f"stderr={(result.stderr or '')[:1200]}"
            )

        return result

    def _fail(self, msg: str) -> None:
        """
        Centralized failure behavior.
        In strict mode: raise (fails CI with non-zero exit).
        Otherwise: just log and return.
        """
        print(msg)
        if self._strict_enabled():
            raise RuntimeError(msg)

    def _mark_blocked(self, milestone_id: str, reason: str) -> None:
        update_milestone_status(self.config.milestones, milestone_id, "blocked", reason)
        self.config.save()

    # -------------------------
    # Core operations
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
                result = subprocess.run(
                    cmd,
                    cwd=str(app_path),
                    capture_output=True,
                    text=True,
                    timeout=300,
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
                    {"name": check["name"], "command": check["command"], "error": "Command timed out", "output": ""}
                )
            except Exception as e:
                failing.append(
                    {"name": check["name"], "command": check["command"], "error": str(e), "output": ""}
                )

        return failing

    def apply_patch(self, patch: str, base_path: Path) -> Tuple[bool, Optional[str]]:
        """Apply unified diff patch to repository."""
        lines = patch.strip().split("\n")
        diff_start = None
        for i, line in enumerate(lines):
            if line.startswith("---") or line.startswith("+++"):
                diff_start = i
                break

        if diff_start is None:
            return False, "No diff found in patch"

        clean_patch = "\n".join(lines[diff_start:])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(clean_patch)
            patch_path = f.name

        try:
            # Check patch
            result = self._run_cmd(["git", "apply", "--check", patch_path], cwd=base_path, timeout=120, check=False, label="git apply --check")
            if result.returncode != 0:
                return False, f"Patch check failed: {(result.stderr or '').strip()}"

            # Apply patch
            result = self._run_cmd(["git", "apply", patch_path], cwd=base_path, timeout=120, check=False, label="git apply")
            if result.returncode != 0:
                return False, f"Patch apply failed: {(result.stderr or '').strip()}"

            return True, None
        finally:
            try:
                os.unlink(patch_path)
            except Exception:
                pass

    # -------------------------
    # Fix mode
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
                traceback.print_exc()
            self._fail(f"OpenAI exception in fix mode: {type(e).__name__}: {e}")
            return None

        if not patch:
            print("Failed to generate patch (empty response in fix mode)")
            self._fail("OpenAI returned empty patch in fix mode")
            return None

        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-fix-truth-checks"
        self.github.create_branch(branch_name, self.config.default_branch)

        # Checkout branch locally
        self._run_cmd(["git", "checkout", "-b", branch_name], cwd=self.repo_path, timeout=120, check=True, label="git checkout -b")

        success, error = self.apply_patch(patch, self.repo_path)
        if not success:
            self._fail(f"Patch apply failed in fix mode: {error}")
            return None

        print("Verifying fixes...")
        still_failing = self.run_truth_checks()
        if still_failing:
            self._fail("Truth checks still failing after fix patch")
            return None

        # Commit + push
        self._run_cmd(["git", "add", "-A"], cwd=self.repo_path, timeout=120, check=True, label="git add")
        self._run_cmd(["git", "commit", "-m", "fix: resolve failing truth checks"], cwd=self.repo_path, timeout=120, check=True, label="git commit")
        self._run_cmd(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, timeout=180, check=True, label="git push")

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

    # -------------------------
    # Milestone mode
    # -------------------------
    def run_milestone_mode(self) -> Optional[str]:
        """Run in milestone mode - complete next todo milestone."""
        milestone = get_next_todo_milestone(self.config.milestones)
        if not milestone:
            print("No todo milestones found")
            return None

        milestone_id = milestone["id"]
        title = milestone.get("title", milestone_id)
        print(f"Processing milestone: {title} ({milestone_id})")

        # Mark in progress
        update_milestone_status(self.config.milestones, milestone_id, "in_progress")
        self.config.save()

        try:
            prompt = build_milestone_prompt(
                milestone,
                self.config.repo_rules,
                self.config.max_files,
                self.config.max_lines,
            )

            print("Calling OpenAI API for milestone patch...")
            patch = None
            try:
                patch = call_openai(prompt, self.config.openai_api_key)
            except Exception as e:
                if self._debug_enabled():
                    print("=== OpenAI exception (milestone mode) ===")
                    traceback.print_exc()
                self._mark_blocked(milestone_id, f"OpenAI exception: {type(e).__name__}: {e}")
                self._fail(f"OpenAI exception in milestone mode: {type(e).__name__}: {e}")
                return None

            if not patch:
                self._mark_blocked(milestone_id, "Failed to generate patch: empty response from call_openai")
                self._fail("OpenAI returned empty patch in milestone mode")
                return None

            # Create branch remote + local
            branch_slug = milestone_id.replace("_", "-")
            branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-{branch_slug}"
            self.github.create_branch(branch_name, self.config.default_branch)

            self._run_cmd(["git", "checkout", "-b", branch_name], cwd=self.repo_path, timeout=120, check=True, label="git checkout -b")

            success, error = self.apply_patch(patch, self.repo_path)
            if not success:
                self._mark_blocked(milestone_id, f"Patch apply failed: {error}")
                self._fail(f"Patch apply failed in milestone mode: {error}")
                return None

            # Verify acceptance
            print("Verifying acceptance criteria...")
            app_path = self.repo_path / "app"
            for cmd in milestone.get("acceptance", []):
                cmd_parts = cmd.split()
                res = self._run_cmd(cmd_parts, cwd=app_path, timeout=600, check=False, label=f"acceptance: {cmd}")
                if res.returncode != 0:
                    self._mark_blocked(milestone_id, f"Acceptance failed: {cmd}")
                    self._fail(f"Acceptance criteria not met: {cmd}")
                    return None

            # Commit + push
            self._run_cmd(["git", "add", "-A"], cwd=self.repo_path, timeout=120, check=True, label="git add")
            self._run_cmd(["git", "commit", "-m", f"feat: {title}"], cwd=self.repo_path, timeout=120, check=True, label="git commit")
            self._run_cmd(["git", "push", "-u", "origin", branch_name], cwd=self.repo_path, timeout=180, check=True, label="git push")

            acceptance_outputs = "\n".join([f"- `{cmd}`: ✅ PASS" for cmd in milestone.get("acceptance", [])])

            pr_body = f"""## Summary
Completed milestone: {title}

## Changes
- Applied LLM-generated patch to complete milestone
- All acceptance criteria now passing

## Verification
{acceptance_outputs}

## Files Changed
See diff for details.
"""

            pr = self.github.create_pr(
                title=f"feat: {title}",
                body=pr_body,
                head=branch_name,
                base=self.config.default_branch,
            )

            if pr:
                print(f"Created PR: {pr['html_url']}")

                update_milestone_status(self.config.milestones, milestone_id, "done")
                self.config.save()
                return pr["html_url"]

            # If PR creation failed for some reason, mark blocked so it isn't stuck.
            self._mark_blocked(milestone_id, "PR creation failed (GitHub API returned no PR)")
            self._fail("Failed to create PR for milestone")
            return None

        except Exception as e:
            # Absolute safety net: never leave milestone in_progress on unexpected errors
            if self._debug_enabled():
                print("=== Unhandled exception in milestone mode ===")
                traceback.print_exc()

            # only overwrite to blocked if it's still in progress
            self._mark_blocked(milestone_id, f"Unhandled exception: {type(e).__name__}: {e}")
            self._fail(f"Unhandled exception in milestone mode: {type(e).__name__}: {e}")
            return None

    # -------------------------
    # Main entry
    # -------------------------
    def run(self, mode: str = "auto") -> Optional[str]:
        """Run agent in specified mode."""
        if not self.repo_path or not self.repo_path.exists():
            print("Repository path not found")
            return None

        # Ensure we're on default branch and up to date
        try:
            self._run_cmd(["git", "checkout", self.config.default_branch], cwd=self.repo_path, timeout=120, check=True, label="git checkout default")
            self._run_cmd(["git", "pull"], cwd=self.repo_path, timeout=180, check=True, label="git pull")
        except Exception as e:
            self._fail(f"Git sync failed on default branch: {type(e).__name__}: {e}")
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