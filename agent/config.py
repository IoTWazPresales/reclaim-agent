"""Configuration loading for Reclaim Agent."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration manager for the agent."""
    
    def __init__(self):
        self.repo_path = os.getenv("RECLAIM_REPO_PATH", "")
        self.repo_name = os.getenv("RECLAIM_REPO", "IoTWazPresales/Reclaim")
        self.default_branch = os.getenv("RECLAIM_DEFAULT_BRANCH", "main")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.github_token = os.getenv("RECLAIM_GH_TOKEN", "")
        
        # Load agent config
        agent_dir = Path(__file__).parent.parent
        default_config_path = agent_dir / "agent_config" / "default.yaml"
        reclaim_config_path = agent_dir / "agent_config" / "reclaim.yaml"
        
        default_config = self._load_yaml(default_config_path)
        reclaim_config = self._load_yaml(reclaim_config_path)

        # Merge configs (reclaim overrides default) with a shallow dict + nested dict merge.
        # Lists (e.g. truth_checks, repo_rules) are taken from reclaim_config when present.
        self.config = self._merge_configs(default_config, reclaim_config)
        self.truth_checks = self.config.get("truth_checks", [])
        self.milestones = self.config.get("milestones") or []
        self.repo_rules = self.config.get("repo_rules", [])
        self.max_files = self.config.get("max_files", 3)
        self.max_lines = self.config.get("max_lines", 150)
        self.max_attempts = self.config.get("max_attempts", 3)
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two config dicts where `override` wins.

        - For plain keys, override value replaces base.
        - For nested dicts, merge recursively.
        - For lists/scalars, override value replaces base.
        """
        result: Dict[str, Any] = dict(base or {})
        for key, val in (override or {}).items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(val, dict)
            ):
                result[key] = Config._merge_configs(result[key], val)
            else:
                result[key] = val
        return result

    def get_milestone_by_id(self, milestone_id: str) -> Optional[Dict[str, Any]]:
        """Get milestone by ID."""
        for milestone in self.milestones:
            if milestone.get("id") == milestone_id:
                return milestone
        return None
    
    def get_next_todo_milestone(self) -> Optional[Dict[str, Any]]:
        """Get the first milestone with status 'todo'."""
        for milestone in self.milestones:
            if milestone.get("status") == "todo":
                return milestone
        return None
    
    def update_milestone_status(self, milestone_id: str, status: str, reason: Optional[str] = None):
        """Update milestone status in config and save to reclaim.yaml."""
        milestone = self.get_milestone_by_id(milestone_id)
        if not milestone:
            return False
        
        milestone["status"] = status
        if reason:
            milestone["reason"] = reason
        
        # Save back to reclaim.yaml
        agent_dir = Path(__file__).parent.parent
        reclaim_config_path = agent_dir / "agent_config" / "reclaim.yaml"
        
        reclaim_config = self._load_yaml(reclaim_config_path)
        reclaim_config["milestones"] = self.milestones
        
        with open(reclaim_config_path, "w", encoding="utf-8") as f:
            yaml.dump(reclaim_config, f, default_flow_style=False, sort_keys=False)
        
        return True
    
    def save(self):
        """Save current config state to reclaim.yaml."""
        agent_dir = Path(__file__).parent.parent
        reclaim_config_path = agent_dir / "agent_config" / "reclaim.yaml"
        
        reclaim_config = self._load_yaml(reclaim_config_path)
        reclaim_config["milestones"] = self.milestones
        
        with open(reclaim_config_path, "w", encoding="utf-8") as f:
            yaml.dump(reclaim_config, f, default_flow_style=False, sort_keys=False)
