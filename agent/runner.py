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
from .knowledge_base import KnowledgeBaseGenerator


class Runner:
    """Main runner for agent operations."""

    def __init__(self, config: Config):
        self.config = config
        self.github = GitHubAPI(config.github_token, config.repo_name)
        self.repo_path = Path(config.repo_path) if config.repo_path else None
        self._knowledge_base: Optional[str] = None

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
    
    def _load_knowledge_base(self) -> str:
        """Load or generate knowledge base."""
        if self._knowledge_base is not None:
            return self._knowledge_base
        
        # Try to load existing knowledge base
        kb_path = self.repo_path / "KNOWLEDGE_BASE.md" if self.repo_path else None
        if kb_path and kb_path.exists():
            try:
                self._knowledge_base = kb_path.read_text(encoding="utf-8")
                if self._debug_enabled():
                    print(f"[AGENT_DEBUG] Loaded knowledge base from {kb_path} ({len(self._knowledge_base)} chars)")
                return self._knowledge_base
            except Exception as e:
                if self._debug_enabled():
                    print(f"[AGENT_DEBUG] Failed to load knowledge base: {e}")
        
        # Generate knowledge base if it doesn't exist
        if self.repo_path:
            try:
                # Use LLM analysis if API key is available (hybrid approach)
                use_llm = bool(self.config.openai_api_key)
                generator = KnowledgeBaseGenerator(
                    str(self.repo_path),
                    openai_api_key=self.config.openai_api_key if use_llm else None,
                    use_llm_analysis=use_llm
                )
                self._knowledge_base = generator.generate()
                # Optionally save it (but don't commit - it's generated)
                if self._debug_enabled():
                    llm_status = "with LLM semantic analysis" if use_llm else "structure-only"
                    print(f"[AGENT_DEBUG] Generated knowledge base {llm_status} ({len(self._knowledge_base)} chars)")
                return self._knowledge_base
            except Exception as e:
                if self._debug_enabled():
                    print(f"[AGENT_DEBUG] Failed to generate knowledge base: {e}")
        
        # Fallback: return empty string
        return ""

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

    def _parse_file_content_format(self, content: str, base_path: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Parse the new file content format (===FILE_START: path=== ... ===FILE_END: path===)
        and generate a unified diff using git diff.
        
        Returns: (success, error_message, unified_diff_patch)
        """
        import re
        
        # Pattern to match file blocks - be flexible with whitespace and path matching
        # Match FILE_START with path, then content, then FILE_END with same path (flexible whitespace)
        file_pattern = re.compile(
            r'===FILE_START:\s*(.+?)\s*===\s*\n(.*?)\n===FILE_END:\s*\1\s*===',
            re.DOTALL | re.MULTILINE
        )
        
        files_to_write = {}
        matches = list(file_pattern.finditer(content))
        
        if not matches:
            return False, "No file blocks found in format ===FILE_START: path=== ... ===FILE_END: path===", None
        
        # Parse all file blocks
        for match in matches:
            file_path = match.group(1).strip()
            file_content = match.group(2)
            
            # Normalize path (remove leading/trailing whitespace, handle relative paths)
            if file_path.startswith("/"):
                file_path = file_path[1:]
            
            files_to_write[file_path] = file_content
        
        if not files_to_write:
            return False, "No valid file blocks found", None
        
        # Write files temporarily, generate diff, then restore
        import tempfile
        import shutil
        
        written_files = []
        backups = {}  # file_path -> backup_path
        
        try:
            # Step 1: Backup existing files and write new content
            for file_path, file_content in files_to_write.items():
                actual_file = base_path / file_path
                
                # Backup existing file if it exists
                if actual_file.exists() and actual_file.is_file():
                    backup_path = base_path / f"{file_path}.agent_backup"
                    shutil.copy2(actual_file, backup_path)
                    backups[file_path] = backup_path
                
                # Write new content
                actual_file.parent.mkdir(parents=True, exist_ok=True)
                actual_file.write_text(file_content, encoding="utf-8")
                written_files.append(file_path)
            
            # Step 2: Stage all files
            for file_path in written_files:
                self._run_cmd(
                    ["git", "add", file_path],
                    cwd=base_path,
                    timeout=30,
                    label=f"git add {file_path}",
                )
            
            # Step 3: Generate unified diff
            diff_result = self._run_cmd(
                ["git", "diff", "--cached", "--no-color"],
                cwd=base_path,
                timeout=60,
                label="git diff --cached",
            )
            
            if diff_result.returncode != 0:
                raise RuntimeError(f"Failed to generate diff: {diff_result.stderr}")
            
            unified_diff = diff_result.stdout or ""
            
            if not unified_diff.strip():
                raise RuntimeError("Generated diff is empty (no changes detected)")
            
            # Step 4: Unstage and restore original files
            self._run_cmd(
                ["git", "reset", "HEAD", "--"] + written_files,
                cwd=base_path,
                timeout=30,
                label="git reset",
            )
            
            # Restore original files from backups
            for file_path, backup_path in backups.items():
                shutil.move(backup_path, base_path / file_path)
            
            # Remove new files that didn't exist before
            for file_path in written_files:
                if file_path not in backups:
                    actual_file = base_path / file_path
                    if actual_file.exists():
                        actual_file.unlink()
            
            return True, None, unified_diff
            
        except Exception as e:
            # Cleanup: restore all backups
            for file_path, backup_path in backups.items():
                try:
                    if backup_path.exists():
                        shutil.move(backup_path, base_path / file_path)
                except:
                    pass
            
            # Remove new files on error
            for file_path in written_files:
                if file_path not in backups:
                    try:
                        actual_file = base_path / file_path
                        if actual_file.exists():
                            actual_file.unlink()
                    except:
                        pass
            
            return False, f"Error processing file content format: {e}", None

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
            
            # Check if file exists (unless it's a new file indicated by /dev/null)
            if header_path != "/dev/null":
                actual_file = base_path / header_path
                if not actual_file.exists():
                    return False, (
                        f"Patch targets file '{header_path}' which does not exist in the repository. "
                        f"Use REAL file paths from the repository. If creating a new file, use '--- /dev/null'."
                    )
                elif not actual_file.is_file():
                    return False, (
                        f"Patch targets '{header_path}' which exists but is not a regular file."
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
                # Extract line number from error if present (e.g., "error: corrupt patch at line 20")
                error_msg = check.stderr or ""
                import re
                
                # Enhanced debugging: extract hunk info and compare with actual file
                line_num_match = re.search(r"line (\d+)", error_msg)
                if line_num_match:
                    try:
                        patch_line_num = int(line_num_match.group(1))
                        patch_lines = clean_patch.split("\n")
                        if 0 < patch_line_num <= len(patch_lines):
                            problematic_patch_line = patch_lines[patch_line_num - 1]
                            error_msg = f"{error_msg}\nProblematic patch line {patch_line_num}: {problematic_patch_line[:200]}"
                            
                            # Try to extract file path and hunk info for better debugging
                            hunk_match = re.search(r"^@@\s+-(\d+),(\d+)\s+\+(\d+),(\d+)\s+@@", problematic_patch_line)
                            if hunk_match:
                                old_start = int(hunk_match.group(1))
                                old_count = int(hunk_match.group(2))
                                
                                # Find the file path from the patch
                                file_path = None
                                for i in range(patch_line_num - 1, -1, -1):
                                    if i < len(patch_lines):
                                        line = patch_lines[i]
                                        if line.startswith("--- "):
                                            path_part = line[4:].strip()
                                            if path_part.startswith("a/"):
                                                path_part = path_part[2:]
                                            if path_part != "/dev/null":
                                                file_path = path_part
                                                break
                                
                                # If we found the file, show what's actually there
                                if file_path:
                                    actual_file = base_path / file_path
                                    if actual_file.exists() and actual_file.is_file():
                                        try:
                                            file_content = actual_file.read_text(encoding="utf-8")
                                            file_lines = file_content.split("\n")
                                            # Show context around the problematic line
                                            context_start = max(0, old_start - 5)
                                            context_end = min(len(file_lines), old_start + old_count + 5)
                                            actual_context = "\n".join([
                                                f"{i+1:4d}: {line}" 
                                                for i, line in enumerate(file_lines[context_start:context_end], start=context_start)
                                            ])
                                            error_msg += f"\n\nActual file content around line {old_start} in {file_path}:\n{actual_context}"
                                            
                                            # Show what the patch expects
                                            patch_context_lines = []
                                            in_hunk = False
                                            for i, line in enumerate(patch_lines):
                                                if line.startswith("@@") and i < patch_line_num:
                                                    in_hunk = True
                                                    patch_context_lines = []
                                                elif in_hunk and (line.startswith(" ") or line.startswith("-") or line.startswith("+")):
                                                    patch_context_lines.append(line)
                                                elif line.startswith("@@") and i >= patch_line_num:
                                                    break
                                            
                                            if patch_context_lines:
                                                error_msg += f"\n\nWhat the patch expects (first 10 context lines):\n" + "\n".join(patch_context_lines[:10])
                                        except Exception as e:
                                            if self._debug_enabled():
                                                error_msg += f"\n[Could not read file for debugging: {e}]"
                    except (ValueError, IndexError) as e:
                        if self._debug_enabled():
                            error_msg += f"\n[Error extracting debug info: {e}]"
                
                return False, f"Patch check failed: {error_msg}"

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

        # Check for NO_PATCH escape hatch
        if patch.strip() == "NO_PATCH":
            print("Model declined to generate patch (NO_PATCH response)")
            self._fail("Model unable to safely generate a patch for this fix")
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

        # Gather context for the LLM using knowledge base + targeted file reading
        current_files_snippet: Optional[str] = None
        try:
            if self.repo_path:
                context_parts: List[str] = []
                
                # 0. Load knowledge base (comprehensive codebase understanding)
                kb_content = self._load_knowledge_base()
                if kb_content:
                    context_parts.append("=== KNOWLEDGE BASE (Complete Codebase Understanding) ===")
                    context_parts.append(kb_content)
                    context_parts.append("=== END KNOWLEDGE BASE ===\n")
                    
                    # When KB is available, skip redundant structure gathering
                    # KB already has: structure, file catalog, patterns, navigation
                    # We only need: target files to modify
                    skip_redundant_gathering = True
                else:
                    skip_redundant_gathering = False
                
                # 1. Get directory structure overview (only if no KB - KB has this)
                if not skip_redundant_gathering and milestone.get("target_files"):
                    structure_parts: List[str] = []
                    all_files_for_structure: List[str] = []
                    
                    # First, collect all files matching patterns
                    for pattern in milestone["target_files"]:
                        result = self._run_cmd(
                            ["git", "ls-files", pattern],
                            cwd=self.repo_path,
                            timeout=60,
                            label=f"git ls-files structure {pattern}",
                        )
                        if result.returncode == 0:
                            for rel_path in (result.stdout or "").splitlines():
                                if rel_path.strip():
                                    all_files_for_structure.append(rel_path.strip())
                    
                    if all_files_for_structure:
                        # Build a tree structure from file paths
                        dirs_seen = set()
                        files_by_dir: Dict[str, List[str]] = {}
                        
                        for file_path in all_files_for_structure:
                            parts = file_path.split("/")
                            filename = parts[-1]
                            dir_path = "/".join(parts[:-1]) if len(parts) > 1 else ""
                            
                            # Track all parent directories
                            for i in range(1, len(parts)):
                                parent_dir = "/".join(parts[:i])
                                dirs_seen.add(parent_dir)
                            
                            # Group files by directory
                            if dir_path not in files_by_dir:
                                files_by_dir[dir_path] = []
                            files_by_dir[dir_path].append(filename)
                        
                        # Build tree output
                        structure_parts.append("REPOSITORY STRUCTURE (target areas):")
                        # Show root-level files first
                        if "" in files_by_dir:
                            structure_parts.append("  [root]/")
                            for f in sorted(files_by_dir[""])[:20]:
                                structure_parts.append(f"    {f}")
                        
                        # Then show directories in sorted order
                        sorted_dirs = sorted([d for d in dirs_seen if d])
                        for d in sorted_dirs[:30]:  # Limit to 30 directories
                            depth = d.count("/")
                            indent = "  " + ("  " * depth)
                            structure_parts.append(f"{indent}{d.split('/')[-1]}/")
                            # Show files in this directory
                            if d in files_by_dir:
                                for f in sorted(files_by_dir[d])[:15]:  # Max 15 files per dir
                                    structure_parts.append(f"{indent}  {f}")
                    
                    if structure_parts:
                        context_parts.append("\n".join(structure_parts))
                
                # 2. Get target files list (simplified when KB available)
                all_matching_files: List[str] = []
                if milestone.get("target_files"):
                    for pattern in milestone["target_files"]:
                        result = self._run_cmd(
                            ["git", "ls-files", pattern],
                            cwd=self.repo_path,
                            timeout=60,
                            label=f"git ls-files {pattern}",
                        )
                        if result.returncode == 0:
                            for rel_path in (result.stdout or "").splitlines():
                                if rel_path.strip():
                                    all_matching_files.append(rel_path.strip())
                
                # Only show file list if no KB (KB has comprehensive file catalog)
                if not skip_redundant_gathering and all_matching_files:
                    context_parts.append(f"\nALL FILES MATCHING TARGET PATTERNS ({len(all_matching_files)} files):")
                    for f in sorted(all_matching_files)[:200]:  # Limit to 200 files for context
                        context_parts.append(f"  - {f}")
                
                # 3. Include key config files (only if no KB - KB has config info)
                if not skip_redundant_gathering:
                    config_files = [
                        "package.json",
                        "tsconfig.json",
                        "app/package.json",
                        "app/tsconfig.json",
                    ]
                    config_content = []
                    for config_path in config_files:
                        full_path = self.repo_path / config_path
                        if full_path.exists() and full_path.is_file():
                            try:
                                content = full_path.read_text(encoding="utf-8")
                                # Limit config file size to 5000 chars
                                if len(content) > 5000:
                                    content = content[:5000] + "\n... [truncated]"
                                config_content.append(f"--- CONFIG: {config_path} ---\n{content}\n")
                            except Exception:
                                pass
                    
                    if config_content:
                        context_parts.append("\nKEY CONFIGURATION FILES:\n" + "\n".join(config_content))
                
                # 4. Gather file content for target files (ALWAYS needed - these are the files to modify)
                # When KB is available, we can be more selective and focused
                if milestone.get("target_files") and all_matching_files:
                    snippets: List[str] = []
                    if skip_redundant_gathering:
                        # With KB: focus on fewer, most relevant files
                        max_files_full = 3  # Top 3 files get FULL content
                        max_files_total = 5  # Only 5 files total (KB provides context)
                        max_chars_per_file_truncated = 30000  # Smaller truncation limit
                        max_total_chars = 100000  # Smaller total budget (~250K tokens)
                    else:
                        # Without KB: need more files for context
                        max_files_full = 3  # Top 3 files get FULL content
                        max_files_total = 10  # More files needed
                        max_chars_per_file_truncated = 50000
                        max_total_chars = 200000  # Larger budget (~500K tokens)
                    matched = 0
                    total_chars = 0
                    
                    # Prioritize files that are most likely relevant based on milestone title
                    title_lower = milestone.get("title", "").lower()
                    priority_keywords = []
                    if "training" in title_lower:
                        priority_keywords.extend(["training", "Training"])
                    if "screen" in title_lower or "ui" in title_lower or "panel" in title_lower:
                        priority_keywords.extend(["Screen", "screen", "component", "Component"])
                    if "preview" in title_lower:
                        priority_keywords.extend(["preview", "Preview"])
                    
                    # Always prioritize types.ts files - they're critical for type safety
                    def priority_score(path: str) -> int:
                        score = 0
                        # Types files get highest priority
                        if path.endswith("types.ts") or "/types.ts" in path:
                            score += 50
                        for keyword in priority_keywords:
                            if keyword in path:
                                score += 10
                        return score
                    
                    all_matching_files.sort(key=priority_score, reverse=True)
                    
                    # Read files: full content for top priority, truncated for others
                    for idx, rel_path in enumerate(all_matching_files[:max_files_total]):
                        if total_chars >= max_total_chars:
                            break
                        file_path = (self.repo_path / rel_path)
                        if not file_path.is_file():
                            continue
                        try:
                            text = file_path.read_text(encoding="utf-8")
                            file_lines = text.count('\n') + (1 if text else 0)
                            
                            # Top priority files get FULL content (no truncation)
                            is_full_file = idx < max_files_full
                            
                            if is_full_file:
                                # Full file content - critical for accurate line numbers
                                file_info = f"--- FILE: {rel_path} (FULL CONTENT, {file_lines} lines, {len(text)} chars) ---\n"
                                snippet = text
                                chars_to_take = len(text)
                            else:
                                # Truncated content for lower priority files
                                remaining_budget = max_total_chars - total_chars
                                chars_to_take = min(max_chars_per_file_truncated, remaining_budget, len(text))
                                snippet = text[:chars_to_take]
                                
                                file_info = f"--- FILE: {rel_path} (PARTIAL: first {chars_to_take} of {len(text)} chars, {file_lines} total lines) ---\n"
                                if chars_to_take < len(text):
                                    snippet += f"\n... [file truncated, {len(text) - chars_to_take} more chars remain]"
                            
                            snippets.append(file_info + snippet + "\n")
                            matched += 1
                            total_chars += chars_to_take
                            
                            if matched >= max_files_total or total_chars >= max_total_chars:
                                break
                        except Exception:
                            continue
                    
                    if snippets:
                        full_count = min(max_files_full, len(snippets))
                        context_parts.append(
                            f"\nFILE CONTENTS ({full_count} full files + {len(snippets) - full_count} partial files):\n" + "\n".join(snippets)
                        )
                
                if context_parts:
                    current_files_snippet = "\n\n".join(context_parts)
                    if self._debug_enabled():
                        print(f"[AGENT_DEBUG] Gathered comprehensive context: {len(context_parts)} sections, ~{len(current_files_snippet)} chars")
        except Exception as e:
            # Context gathering should never break the run.
            if self._debug_enabled():
                print(f"[AGENT_DEBUG] File context gathering failed: {e}")
                import traceback
                traceback.print_exc()
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

        # Check for NO_PATCH escape hatch
        if patch.strip() == "NO_PATCH":
            print("Model declined to generate patch (NO_PATCH response)")
            update_milestone_status(
                self.config.milestones,
                milestone_id,
                "blocked",
                "Model unable to safely generate a patch - insufficient context or unclear requirements"
            )
            self.config.save()
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

        # Try new file content format first (===FILE_START: path=== ... ===FILE_END: path===)
        # If that fails, check if it's unified diff format and reject it with helpful error
        actual_patch = patch
        parse_success, parse_error, generated_diff = self._parse_file_content_format(patch, self.repo_path)
        
        # Extract file paths from patch (either from file content format or unified diff)
        patch_file_paths = set()
        if parse_success and generated_diff:
            # Extract from generated unified diff
            for line in generated_diff.split("\n"):
                if line.startswith("--- ") or line.startswith("+++ "):
                    path_part = line[4:].strip()
                    if path_part.startswith("a/"):
                        path_part = path_part[2:]
                    elif path_part.startswith("b/"):
                        path_part = path_part[2:]
                    if path_part != "/dev/null":
                        patch_file_paths.add(path_part)
        else:
            # Try to extract from file content format
            import re
            file_pattern = re.compile(r'===FILE_START:\s*(.+?)===', re.MULTILINE)
            for match in file_pattern.finditer(patch):
                file_path = match.group(1).strip()
                if file_path.startswith("/"):
                    file_path = file_path[1:]
                patch_file_paths.add(file_path)
            
            # If no file content format found, try unified diff format
            if not patch_file_paths:
                for line in patch.split("\n"):
                    if line.startswith("--- ") or line.startswith("+++ "):
                        path_part = line[4:].strip()
                        if path_part.startswith("a/"):
                            path_part = path_part[2:]
                        elif path_part.startswith("b/"):
                            path_part = path_part[2:]
                        if path_part != "/dev/null":
                            patch_file_paths.add(path_part)
        
        # Validate that patch touches at least one target file (if target_files are specified)
        if milestone.get("target_files") and patch_file_paths:
            # Get all allowed target files
            allowed_files = set()
            for pattern in milestone["target_files"]:
                result = self._run_cmd(
                    ["git", "ls-files", pattern],
                    cwd=self.repo_path,
                    timeout=60,
                    label=f"git ls-files validation {pattern}",
                )
                if result.returncode == 0:
                    for rel_path in (result.stdout or "").splitlines():
                        if rel_path.strip():
                            allowed_files.add(rel_path.strip())

            # Check if any patch file matches allowed files
            if allowed_files:
                matches = patch_file_paths.intersection(allowed_files)
                if not matches:
                    # None of the files in the patch are in the allowed target_files
                    update_milestone_status(
                        self.config.milestones,
                        milestone_id,
                        "blocked",
                        f"Patch touches files outside target_files: {list(patch_file_paths)[:3]}. Must only modify files matching {milestone['target_files']}"
                    )
                    self.config.save()
                    self._fail(f"Patch does not touch any allowed target files. Patch files: {list(patch_file_paths)}, Allowed patterns: {milestone['target_files']}")
                    return None
        if parse_success and generated_diff:
            print("Successfully parsed file content format, using generated diff")
            actual_patch = generated_diff
        else:
            # Check if model output unified diff format (which we don't want)
            if patch.strip().startswith("--- ") or "--- a/" in patch or "+++ b/" in patch:
                error_msg = (
                    "Model output unified diff format instead of file content format. "
                    "The model must output complete file content using ===FILE_START: path=== ... ===FILE_END: path=== format. "
                    "Unified diff format with line numbers is not supported and causes corrupt patch errors."
                )
                update_milestone_status(
                    self.config.milestones,
                    milestone_id,
                    "blocked",
                    error_msg
                )
                self.config.save()
                self._fail(error_msg)
                return None
            
            # Not in new format and not unified diff - might be some other issue
            print(f"File content format parse failed: {parse_error}, checking if it's a different format...")
            # Fall through to try unified diff as last resort (though it will likely fail)

        success, error = self.apply_patch(actual_patch, self.repo_path)
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
