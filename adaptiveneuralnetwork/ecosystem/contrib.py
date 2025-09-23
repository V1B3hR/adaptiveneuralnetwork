"""
Community contribution system and governance.
"""

import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum

from .plugins import PluginBase, PluginMetadata


class ContributionType(Enum):
    """Types of community contributions."""
    PLUGIN = "plugin"
    MODEL = "model"
    DATASET = "dataset"
    DOCUMENTATION = "documentation"
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    OPTIMIZATION = "optimization"


class ContributionStatus(Enum):
    """Status of a contribution."""
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"
    DEPRECATED = "deprecated"


@dataclass
class Contributor:
    """Information about a contributor."""
    id: str
    name: str
    email: str
    github_username: Optional[str] = None
    affiliation: Optional[str] = None
    contributions_count: int = 0
    reputation_score: float = 0.0
    joined_date: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "github_username": self.github_username,
            "affiliation": self.affiliation,
            "contributions_count": self.contributions_count,
            "reputation_score": self.reputation_score,
            "joined_date": self.joined_date.isoformat(),
            "last_active": self.last_active.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contributor':
        """Create from dictionary."""
        data = data.copy()
        if "joined_date" in data:
            data["joined_date"] = datetime.fromisoformat(data["joined_date"])
        if "last_active" in data:
            data["last_active"] = datetime.fromisoformat(data["last_active"])
        return cls(**data)


@dataclass
class Contribution:
    """Information about a contribution."""
    id: str
    title: str
    description: str
    contributor_id: str
    contribution_type: ContributionType
    status: ContributionStatus = ContributionStatus.SUBMITTED
    submitted_date: datetime = field(default_factory=datetime.utcnow)
    review_date: Optional[datetime] = None
    merge_date: Optional[datetime] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    review_comments: List[Dict[str, Any]] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "contributor_id": self.contributor_id,
            "contribution_type": self.contribution_type.value,
            "status": self.status.value,
            "submitted_date": self.submitted_date.isoformat(),
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "merge_date": self.merge_date.isoformat() if self.merge_date else None,
            "version": self.version,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "review_comments": self.review_comments,
            "test_results": self.test_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contribution':
        """Create from dictionary."""
        data = data.copy()
        data["contribution_type"] = ContributionType(data["contribution_type"])
        data["status"] = ContributionStatus(data["status"])
        data["submitted_date"] = datetime.fromisoformat(data["submitted_date"])
        if data.get("review_date"):
            data["review_date"] = datetime.fromisoformat(data["review_date"])
        if data.get("merge_date"):
            data["merge_date"] = datetime.fromisoformat(data["merge_date"])
        return cls(**data)


class CommunityPlugin(PluginBase):
    """Base class for community-contributed plugins."""
    
    def __init__(self, contribution: Contribution):
        super().__init__()
        self.contribution = contribution
        self._metadata = None
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata from contribution."""
        if self._metadata is None:
            self._metadata = PluginMetadata(
                name=self.contribution.title.lower().replace(" ", "_"),
                version=self.contribution.version,
                author=self.contribution.contributor_id,
                description=self.contribution.description,
                dependencies=self.contribution.dependencies,
                tags=self.contribution.tags
            )
        return self._metadata
    
    def get_contribution_info(self) -> Contribution:
        """Get contribution information."""
        return self.contribution


class ContributionValidator:
    """Validator for community contributions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = {
            ContributionType.PLUGIN: self._validate_plugin,
            ContributionType.MODEL: self._validate_model,
            ContributionType.DATASET: self._validate_dataset,
            ContributionType.DOCUMENTATION: self._validate_documentation,
            ContributionType.BUG_FIX: self._validate_bug_fix,
            ContributionType.FEATURE: self._validate_feature,
            ContributionType.OPTIMIZATION: self._validate_optimization
        }
    
    def validate_contribution(self, contribution: Contribution, 
                            file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a contribution."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "score": 0.0
        }
        
        # Basic validation
        if not contribution.title or len(contribution.title) < 5:
            validation_result["errors"].append("Title must be at least 5 characters long")
        
        if not contribution.description or len(contribution.description) < 20:
            validation_result["errors"].append("Description must be at least 20 characters long")
        
        if not contribution.contributor_id:
            validation_result["errors"].append("Contributor ID is required")
        
        # Type-specific validation
        validator = self.validation_rules.get(contribution.contribution_type)
        if validator:
            type_result = validator(contribution, file_content)
            validation_result["errors"].extend(type_result.get("errors", []))
            validation_result["warnings"].extend(type_result.get("warnings", []))
            validation_result["suggestions"].extend(type_result.get("suggestions", []))
        
        # Calculate score
        validation_result["score"] = self._calculate_score(contribution, validation_result)
        
        # Determine if valid
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    def _validate_plugin(self, contribution: Contribution, 
                        file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a plugin contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        if not contribution.dependencies:
            result["warnings"].append("No dependencies specified - consider adding them")
        
        if not contribution.tags:
            result["warnings"].append("No tags specified - consider adding descriptive tags")
        
        if file_content:
            # Check if it's valid Python code
            try:
                code = file_content.decode('utf-8')
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                result["errors"].append(f"Python syntax error: {e}")
            except UnicodeDecodeError:
                result["errors"].append("File must be valid UTF-8 encoded Python code")
            
            # Check for required imports
            if b"PluginBase" not in file_content:
                result["errors"].append("Plugin must inherit from PluginBase")
            
            # Check for required methods
            required_methods = [b"get_metadata", b"initialize", b"finalize"]
            for method in required_methods:
                if method not in file_content:
                    result["errors"].append(f"Plugin must implement {method.decode()} method")
        
        return result
    
    def _validate_model(self, contribution: Contribution, 
                       file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a model contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check metadata
        if not contribution.metadata.get("model_type"):
            result["errors"].append("Model type must be specified in metadata")
        
        if not contribution.metadata.get("input_shape"):
            result["warnings"].append("Input shape should be specified in metadata")
        
        if not contribution.metadata.get("output_shape"):
            result["warnings"].append("Output shape should be specified in metadata")
        
        if not contribution.metadata.get("performance_metrics"):
            result["warnings"].append("Performance metrics should be provided")
        
        return result
    
    def _validate_dataset(self, contribution: Contribution, 
                         file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a dataset contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        # Check metadata
        if not contribution.metadata.get("size"):
            result["warnings"].append("Dataset size should be specified")
        
        if not contribution.metadata.get("format"):
            result["errors"].append("Dataset format must be specified")
        
        if not contribution.metadata.get("license"):
            result["warnings"].append("Dataset license should be specified")
        
        return result
    
    def _validate_documentation(self, contribution: Contribution, 
                              file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a documentation contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        if file_content:
            content = file_content.decode('utf-8', errors='ignore')
            
            # Check for basic markdown structure
            if not any(line.startswith('#') for line in content.split('\n')):
                result["warnings"].append("Documentation should include headers")
            
            # Check for examples
            if 'example' not in content.lower():
                result["suggestions"].append("Consider adding examples to documentation")
        
        return result
    
    def _validate_bug_fix(self, contribution: Contribution, 
                         file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a bug fix contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        if not contribution.metadata.get("bug_report_id"):
            result["warnings"].append("Bug report ID should be referenced")
        
        if not contribution.metadata.get("affected_versions"):
            result["warnings"].append("Affected versions should be specified")
        
        return result
    
    def _validate_feature(self, contribution: Contribution, 
                         file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a feature contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        if not contribution.metadata.get("use_cases"):
            result["warnings"].append("Use cases should be described")
        
        if not contribution.metadata.get("api_changes"):
            result["warnings"].append("API changes should be documented")
        
        return result
    
    def _validate_optimization(self, contribution: Contribution, 
                             file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate an optimization contribution."""
        result = {"errors": [], "warnings": [], "suggestions": []}
        
        if not contribution.metadata.get("performance_improvement"):
            result["errors"].append("Performance improvement metrics must be provided")
        
        if not contribution.metadata.get("benchmark_results"):
            result["warnings"].append("Benchmark results should be provided")
        
        return result
    
    def _calculate_score(self, contribution: Contribution, 
                        validation_result: Dict[str, Any]) -> float:
        """Calculate quality score for a contribution."""
        score = 100.0
        
        # Deduct for errors
        score -= len(validation_result["errors"]) * 20
        
        # Deduct for warnings
        score -= len(validation_result["warnings"]) * 5
        
        # Bonus for good practices
        if contribution.dependencies:
            score += 5
        
        if contribution.tags:
            score += 5
        
        if contribution.metadata.get("tests"):
            score += 10
        
        if contribution.metadata.get("documentation"):
            score += 10
        
        return max(0.0, min(100.0, score))


class ContributionManager:
    """Manager for community contributions."""
    
    def __init__(self, storage_path: str = "contributions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.contributors_file = self.storage_path / "contributors.json"
        self.contributions_file = self.storage_path / "contributions.json"
        
        self.contributors = self._load_contributors()
        self.contributions = self._load_contributions()
        self.validator = ContributionValidator()
        self.logger = logging.getLogger(__name__)
    
    def _load_contributors(self) -> Dict[str, Contributor]:
        """Load contributors from storage."""
        if not self.contributors_file.exists():
            return {}
        
        try:
            with open(self.contributors_file) as f:
                data = json.load(f)
            
            return {
                contributor_id: Contributor.from_dict(contributor_data)
                for contributor_id, contributor_data in data.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to load contributors: {e}")
            return {}
    
    def _save_contributors(self):
        """Save contributors to storage."""
        try:
            data = {
                contributor_id: contributor.to_dict()
                for contributor_id, contributor in self.contributors.items()
            }
            
            with open(self.contributors_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save contributors: {e}")
    
    def _load_contributions(self) -> Dict[str, Contribution]:
        """Load contributions from storage."""
        if not self.contributions_file.exists():
            return {}
        
        try:
            with open(self.contributions_file) as f:
                data = json.load(f)
            
            return {
                contribution_id: Contribution.from_dict(contribution_data)
                for contribution_id, contribution_data in data.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to load contributions: {e}")
            return {}
    
    def _save_contributions(self):
        """Save contributions to storage."""
        try:
            data = {
                contribution_id: contribution.to_dict()
                for contribution_id, contribution in self.contributions.items()
            }
            
            with open(self.contributions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save contributions: {e}")
    
    def register_contributor(self, name: str, email: str, 
                           github_username: Optional[str] = None,
                           affiliation: Optional[str] = None) -> str:
        """Register a new contributor."""
        contributor_id = hashlib.md5(f"{name}:{email}".encode()).hexdigest()[:12]
        
        if contributor_id in self.contributors:
            return contributor_id
        
        contributor = Contributor(
            id=contributor_id,
            name=name,
            email=email,
            github_username=github_username,
            affiliation=affiliation
        )
        
        self.contributors[contributor_id] = contributor
        self._save_contributors()
        
        self.logger.info(f"Registered new contributor: {name} ({contributor_id})")
        return contributor_id
    
    def submit_contribution(self, title: str, description: str, contributor_id: str,
                          contribution_type: ContributionType, file_path: Optional[str] = None,
                          version: str = "1.0.0", tags: Optional[List[str]] = None,
                          dependencies: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a new contribution."""
        if contributor_id not in self.contributors:
            raise ValueError("Contributor not found")
        
        contribution_id = hashlib.md5(f"{title}:{contributor_id}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        
        contribution = Contribution(
            id=contribution_id,
            title=title,
            description=description,
            contributor_id=contributor_id,
            contribution_type=contribution_type,
            file_path=file_path,
            version=version,
            tags=tags or [],
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.contributions[contribution_id] = contribution
        self._save_contributions()
        
        # Update contributor stats
        self.contributors[contributor_id].contributions_count += 1
        self.contributors[contributor_id].last_active = datetime.utcnow()
        self._save_contributors()
        
        self.logger.info(f"Submitted contribution: {title} ({contribution_id})")
        return contribution_id
    
    def review_contribution(self, contribution_id: str, reviewer_id: str,
                          approved: bool, comments: str) -> bool:
        """Review a contribution."""
        if contribution_id not in self.contributions:
            return False
        
        contribution = self.contributions[contribution_id]
        
        # Add review comment
        review_comment = {
            "reviewer_id": reviewer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "approved": approved,
            "comments": comments
        }
        
        contribution.review_comments.append(review_comment)
        contribution.review_date = datetime.utcnow()
        
        # Update status
        if approved:
            contribution.status = ContributionStatus.APPROVED
        else:
            contribution.status = ContributionStatus.REJECTED
        
        self._save_contributions()
        
        # Update contributor reputation
        if approved:
            self.contributors[contribution.contributor_id].reputation_score += 10
        else:
            self.contributors[contribution.contributor_id].reputation_score -= 2
        
        self._save_contributors()
        
        return True
    
    def validate_contribution(self, contribution_id: str, 
                            file_content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate a contribution."""
        if contribution_id not in self.contributions:
            return {"valid": False, "errors": ["Contribution not found"]}
        
        contribution = self.contributions[contribution_id]
        return self.validator.validate_contribution(contribution, file_content)
    
    def get_contribution_stats(self) -> Dict[str, Any]:
        """Get contribution statistics."""
        total_contributions = len(self.contributions)
        status_counts = {}
        type_counts = {}
        
        for contribution in self.contributions.values():
            status = contribution.status.value
            contrib_type = contribution.contribution_type.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[contrib_type] = type_counts.get(contrib_type, 0) + 1
        
        return {
            "total_contributions": total_contributions,
            "total_contributors": len(self.contributors),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "top_contributors": sorted(
                self.contributors.values(),
                key=lambda c: c.contributions_count,
                reverse=True
            )[:10]
        }
    
    def list_contributions(self, status: Optional[ContributionStatus] = None,
                         contribution_type: Optional[ContributionType] = None,
                         contributor_id: Optional[str] = None) -> List[Contribution]:
        """List contributions with optional filters."""
        contributions = list(self.contributions.values())
        
        if status:
            contributions = [c for c in contributions if c.status == status]
        
        if contribution_type:
            contributions = [c for c in contributions if c.contribution_type == contribution_type]
        
        if contributor_id:
            contributions = [c for c in contributions if c.contributor_id == contributor_id]
        
        return sorted(contributions, key=lambda c: c.submitted_date, reverse=True)
    
    def get_contributor(self, contributor_id: str) -> Optional[Contributor]:
        """Get contributor information."""
        return self.contributors.get(contributor_id)
    
    def get_contribution(self, contribution_id: str) -> Optional[Contribution]:
        """Get contribution information."""
        return self.contributions.get(contribution_id)