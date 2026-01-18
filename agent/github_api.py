"""GitHub API interactions for Reclaim Agent."""

import os
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime


class GitHubAPI:
    """GitHub API client for agent operations."""
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
    
    def create_branch(self, branch_name: str, base_branch: str) -> bool:
        """Create a new branch from base branch."""
        # Get SHA of base branch
        ref_url = f"{self.base_url}/repos/{self.repo}/git/ref/heads/{base_branch}"
        response = requests.get(ref_url, headers=self.headers)
        if response.status_code != 200:
            return False
        
        sha = response.json()["object"]["sha"]
        
        # Create new branch
        create_url = f"{self.base_url}/repos/{self.repo}/git/refs"
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": sha
        }
        response = requests.post(create_url, headers=self.headers, json=data)
        return response.status_code in [201, 422]  # 422 means branch already exists
    
    def create_pr(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "main"
    ) -> Optional[Dict[str, Any]]:
        """Create a pull request."""
        url = f"{self.base_url}/repos/{self.repo}/pulls"
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            return response.json()
        return None
    
    def get_issue_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get issue by title (exact match)."""
        url = f"{self.base_url}/repos/{self.repo}/issues"
        params = {"state": "all", "per_page": 100}
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            return None
        
        issues = response.json()
        for issue in issues:
            if issue["title"] == title and "pull_request" not in issue:
                return issue
        return None
    
    def create_or_update_issue(
        self,
        title: str,
        body: str
    ) -> Optional[Dict[str, Any]]:
        """Create or update issue with matching title."""
        # Try to find existing issue
        existing = self.get_issue_by_title(title)
        
        if existing:
            # Update existing issue
            issue_number = existing["number"]
            url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}"
            data = {"body": body}
            response = requests.patch(url, headers=self.headers, json=data)
            if response.status_code == 200:
                return response.json()
        else:
            # Create new issue
            url = f"{self.base_url}/repos/{self.repo}/issues"
            data = {"title": title, "body": body}
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                return response.json()
        
        return None
    
    def get_workflow_runs(
        self,
        workflow_id: Optional[str] = None,
        branch: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent workflow runs."""
        url = f"{self.base_url}/repos/{self.repo}/actions/runs"
        params = {"per_page": limit}
        if branch:
            params["branch"] = branch
        if workflow_id:
            params["workflow_id"] = workflow_id
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json().get("workflow_runs", [])
        return []
    
    def get_pr_by_branch(self, branch: str) -> Optional[Dict[str, Any]]:
        """Get PR by head branch name."""
        url = f"{self.base_url}/repos/{self.repo}/pulls"
        params = {"head": f"{self.repo.split('/')[0]}:{branch}", "state": "all"}
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            prs = response.json()
            if prs:
                return prs[0]
        return None
