"""Milestone management utilities."""

from typing import Dict, Any, List, Optional
from datetime import datetime


def get_next_todo_milestone(milestones: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the first milestone with status 'todo'."""
    for milestone in milestones:
        if milestone.get("status") == "todo":
            return milestone
    return None


def update_milestone_status(
    milestones: List[Dict[str, Any]],
    milestone_id: str,
    status: str,
    reason: Optional[str] = None
) -> bool:
    """Update milestone status in list."""
    for milestone in milestones:
        if milestone.get("id") == milestone_id:
            milestone["status"] = status
            if reason:
                milestone["reason"] = reason
            if status == "in_progress":
                milestone["started_at"] = datetime.now().isoformat()
            elif status in ["done", "blocked"]:
                milestone["completed_at"] = datetime.now().isoformat()
            return True
    return False


def get_milestone_by_id(milestones: List[Dict[str, Any]], milestone_id: str) -> Optional[Dict[str, Any]]:
    """Get milestone by ID."""
    for milestone in milestones:
        if milestone.get("id") == milestone_id:
            return milestone
    return None


def get_milestones_by_status(milestones: List[Dict[str, Any]], status: str) -> List[Dict[str, Any]]:
    """Get all milestones with given status."""
    return [m for m in milestones if m.get("status") == status]
