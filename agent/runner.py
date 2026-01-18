"""Main orchestration logic for the agent."""

import os
import subprocess
import tempfile
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
    
    def run_truth_checks(self) -> List[Dict[str, Any]]:
        """Run truth checks and return failing checks."""
        if not self.repo_path:
            return []
        
        failing = []
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
                    timeout=300
                )
                if result.returncode != 0:
                    failing.append({
                        "name": check["name"],
                        "command": check["command"],
                        "error": result.stderr[:500],
                        "output": result.stdout[:500]
                    })
            except subprocess.TimeoutExpired:
                failing.append({
                    "name": check["name"],
                    "command": check["command"],
                    "error": "Command timed out",
                    "output": ""
                })
            except Exception as e:
                failing.append({
                    "name": check["name"],
                    "command": check["command"],
                    "error": str(e),
                    "output": ""
                })
        
        return failing
    
    def apply_patch(self, patch: str, base_path: Path) -> Tuple[bool, Optional[str]]:
        """Apply unified diff patch to repository."""
        # Clean patch - remove markdown fences and extract diff
        lines = patch.strip().split("\n")
        diff_start = None
        for i, line in enumerate(lines):
            if line.startswith("---") or line.startswith("+++"):
                diff_start = i
                break
        
        if diff_start is None:
            return False, "No diff found in patch"
        
        clean_patch = "\n".join(lines[diff_start:])
        
        # Write patch to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(clean_patch)
            patch_path = f.name
        
        try:
            # Apply patch with git apply
            result = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=str(base_path),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Patch check failed: {result.stderr}"
            
            # Apply patch
            result = subprocess.run(
                ["git", "apply", patch_path],
                cwd=str(base_path),
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Patch apply failed: {result.stderr}"
            
            return True, None
        finally:
            os.unlink(patch_path)
    
    def run_fix_mode(self) -> Optional[str]:
        """Run in fix mode - fix failing truth checks."""
        print("Running truth checks...")
        failing = self.run_truth_checks()
        
        if not failing:
            print("All truth checks passing - no fix needed")
            return None
        
        print(f"Found {len(failing)} failing checks")
        
        # Generate patch using LLM
        prompt = build_fix_prompt(
            failing,
            self.config.repo_rules,
            self.config.max_files,
            self.config.max_lines
        )
        
        print("Calling OpenAI API for fix patch...")
        patch = call_openai(prompt, self.config.openai_api_key)
        
        if not patch:
            print("Failed to generate patch")
            return None
        
        # Create branch
        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-fix-truth-checks"
        self.github.create_branch(branch_name, self.config.default_branch)
        
        # Checkout branch in repo
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        # Apply patch
        success, error = self.apply_patch(patch, self.repo_path)
        
        if not success:
            print(f"Failed to apply patch: {error}")
            return None
        
        # Verify truth checks pass
        print("Verifying fixes...")
        still_failing = self.run_truth_checks()
        
        if still_failing:
            print("Truth checks still failing after patch")
            return None
        
        # Commit changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        subprocess.run(
            ["git", "commit", "-m", "fix: resolve failing truth checks"],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        # Push branch
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        # Create PR
        check_outputs = "\n".join([
            f"- {check['name']}: ✅ PASS"
            for check in self.config.truth_checks
        ])
        
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
            base=self.config.default_branch
        )
        
        if pr:
            print(f"Created PR: {pr['html_url']}")
            return pr['html_url']
        
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
        update_milestone_status(
            self.config.milestones,
            milestone_id,
            "in_progress"
        )
        self.config.save()
        
        # Generate patch using LLM
        prompt = build_milestone_prompt(
            milestone,
            self.config.repo_rules,
            self.config.max_files,
            self.config.max_lines
        )
        
        print("Calling OpenAI API for milestone patch...")
        patch = call_openai(prompt, self.config.openai_api_key)
        
        if not patch:
            print("Failed to generate patch")
            update_milestone_status(
                self.config.milestones,
                milestone_id,
                "blocked",
                "Failed to generate patch"
            )
            self.config.save()
            return None
        
        # Create branch
        branch_slug = milestone_id.replace("_", "-")
        branch_name = f"agent/{datetime.now().strftime('%Y%m%d')}-{branch_slug}"
        self.github.create_branch(branch_name, self.config.default_branch)
        
        # Checkout branch in repo
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        # Apply patch
        success, error = self.apply_patch(patch, self.repo_path)
        
        if not success:
            print(f"Failed to apply patch: {error}")
            update_milestone_status(
                self.config.milestones,
                milestone_id,
                "blocked",
                f"Patch apply failed: {error}"
            )
            self.config.save()
            return None
        
        # Verify acceptance criteria
        print("Verifying acceptance criteria...")
        all_passed = True
        app_path = self.repo_path / "app"
        
        for cmd in milestone.get("acceptance", []):
            cmd_parts = cmd.split()
            result = subprocess.run(
                cmd_parts,
                cwd=str(app_path),
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                all_passed = False
                break
        
        if not all_passed:
            print("Acceptance criteria not met")
            update_milestone_status(
                self.config.milestones,
                milestone_id,
                "blocked",
                "Acceptance criteria not met"
            )
            self.config.save()
            return None
        
        # Commit changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        subprocess.run(
            ["git", "commit", "-m", f"feat: {milestone['title']}"],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        # Push branch
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        # Create PR
        acceptance_outputs = "\n".join([
            f"- `{cmd}`: ✅ PASS"
            for cmd in milestone.get("acceptance", [])
        ])
        
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
            base=self.config.default_branch
        )
        
        if pr:
            print(f"Created PR: {pr['html_url']}")
            
            # Mark milestone as done
            update_milestone_status(
                self.config.milestones,
                milestone_id,
                "done"
            )
            self.config.save()
            
            return pr['html_url']
        
        return None
    
    def run(self, mode: str = "auto") -> Optional[str]:
        """Run agent in specified mode."""
        if not self.repo_path or not self.repo_path.exists():
            print("Repository path not found")
            return None
        
        # Ensure we're on default branch
        subprocess.run(
            ["git", "checkout", self.config.default_branch],
            cwd=str(self.repo_path),
            capture_output=True
        )
        subprocess.run(
            ["git", "pull"],
            cwd=str(self.repo_path),
            capture_output=True
        )
        
        if mode == "fix" or mode == "auto":
            # Check if fixes needed
            failing = self.run_truth_checks()
            if failing:
                return self.run_fix_mode()
        
        if mode == "milestone" or mode == "auto":
            # Check for milestones
            milestone = get_next_todo_milestone(self.config.milestones)
            if milestone:
                result = self.run_milestone_mode()
                if result and milestone.get("stop_feature"):
                    print("Stop feature enabled - stopping after milestone")
                    return result
                return result
        
        print("No work needed - repo is green and no milestones")
        return None
