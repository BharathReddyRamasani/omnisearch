# backend/services/model_registry.py
import os
import json
import time
from typing import Dict, List, Optional
from backend.services.utils import model_dir, safe


class ModelRegistry:
    """Enterprise-grade model registry with versioning and audit trails"""

    REGISTRY_FILE = "model_registry.json"

    @staticmethod
    def get_registry_path() -> str:
        """Get the path to the model registry file"""
        registry_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(registry_dir, exist_ok=True)
        return os.path.join(registry_dir, ModelRegistry.REGISTRY_FILE)

    @staticmethod
    def load_registry() -> Dict:
        """Load the model registry from disk"""
        registry_path = ModelRegistry.get_registry_path()
        if not os.path.exists(registry_path):
            return {"models": {}, "last_updated": None, "version": "1.0"}

        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception:
            # Return empty registry if corrupted
            return {"models": {}, "last_updated": None, "version": "1.0"}

    @staticmethod
    def save_registry(registry: Dict):
        """Save the model registry to disk"""
        registry_path = ModelRegistry.get_registry_path()
        registry["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    @staticmethod
    def register_model(dataset_id: str, metadata: Dict) -> Dict:
        """Register a new model version"""
        registry = ModelRegistry.load_registry()

        if dataset_id not in registry["models"]:
            registry["models"][dataset_id] = {
                "versions": [],
                "active_version": None,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

        # Generate version number
        existing_versions = registry["models"][dataset_id]["versions"]
        version_num = len(existing_versions) + 1
        version_id = f"v{version_num}"

        # Create version entry
        version_entry = {
            "version_id": version_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata,
            "status": "active",
            "model_path": os.path.join(model_dir(dataset_id), "model.pkl"),
            "metadata_path": os.path.join(model_dir(dataset_id), "metadata.json")
        }

        # Add to versions list
        registry["models"][dataset_id]["versions"].append(version_entry)

        # Set as active if it's the first version or better than current active
        if registry["models"][dataset_id]["active_version"] is None:
            registry["models"][dataset_id]["active_version"] = version_id
        else:
            # Compare with current active version
            current_active = ModelRegistry.get_active_version(dataset_id)
            if current_active and metadata.get("best_score", 0) > current_active.get("metadata", {}).get("best_score", 0):
                registry["models"][dataset_id]["active_version"] = version_id

        ModelRegistry.save_registry(registry)

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "version_id": version_id,
            "is_active": registry["models"][dataset_id]["active_version"] == version_id
        }

    @staticmethod
    def get_active_version(dataset_id: str) -> Optional[Dict]:
        """Get the active model version for a dataset"""
        registry = ModelRegistry.load_registry()

        if dataset_id not in registry["models"]:
            return None

        active_version_id = registry["models"][dataset_id]["active_version"]
        if not active_version_id:
            return None

        versions = registry["models"][dataset_id]["versions"]
        for version in versions:
            if version["version_id"] == active_version_id:
                return version

        return None

    @staticmethod
    def get_model_versions(dataset_id: str) -> List[Dict]:
        """Get all versions for a dataset"""
        registry = ModelRegistry.load_registry()

        if dataset_id not in registry["models"]:
            return []

        return registry["models"][dataset_id]["versions"]

    @staticmethod
    def set_active_version(dataset_id: str, version_id: str) -> Dict:
        """Set a specific version as active"""
        registry = ModelRegistry.load_registry()

        if dataset_id not in registry["models"]:
            return {"status": "failed", "error": "Dataset not found in registry"}

        versions = registry["models"][dataset_id]["versions"]
        version_exists = any(v["version_id"] == version_id for v in versions)

        if not version_exists:
            return {"status": "failed", "error": "Version not found"}

        registry["models"][dataset_id]["active_version"] = version_id
        ModelRegistry.save_registry(registry)

        return {"status": "ok", "dataset_id": dataset_id, "active_version": version_id}

    @staticmethod
    def get_registry_summary() -> Dict:
        """Get a summary of all registered models"""
        registry = ModelRegistry.load_registry()

        summary = {
            "total_datasets": len(registry["models"]),
            "total_models": sum(len(model["versions"]) for model in registry["models"].values()),
            "last_updated": registry.get("last_updated"),
            "datasets": {}
        }

        for dataset_id, model_info in registry["models"].items():
            summary["datasets"][dataset_id] = {
                "versions_count": len(model_info["versions"]),
                "active_version": model_info["active_version"],
                "created_at": model_info["created_at"],
                "latest_score": None
            }

            # Get latest version score
            if model_info["versions"]:
                latest_version = max(model_info["versions"], key=lambda x: x["created_at"])
                summary["datasets"][dataset_id]["latest_score"] = latest_version.get("metadata", {}).get("best_score")

        return summary

    @staticmethod
    def validate_model_consistency(dataset_id: str) -> Dict:
        """Validate that registry matches actual files"""
        registry = ModelRegistry.load_registry()

        issues = []

        if dataset_id not in registry["models"]:
            return {"status": "error", "issues": ["Dataset not found in registry"]}

        model_info = registry["models"][dataset_id]
        active_version = model_info["active_version"]

        # Check if active version exists
        versions = model_info["versions"]
        active_version_data = None
        for version in versions:
            if version["version_id"] == active_version:
                active_version_data = version
                break

        if not active_version_data:
            issues.append(f"Active version {active_version} not found in versions list")

        # Check if model file exists
        if active_version_data:
            model_path = active_version_data["model_path"]
            if not os.path.exists(model_path):
                issues.append(f"Model file not found: {model_path}")

            metadata_path = active_version_data["metadata_path"]
            if not os.path.exists(metadata_path):
                issues.append(f"Metadata file not found: {metadata_path}")

        return {
            "status": "ok" if not issues else "warning",
            "issues": issues,
            "active_version": active_version,
            "total_versions": len(versions)
        }


# =====================================================
# REGISTRY INTEGRATION FUNCTIONS
# =====================================================

def register_trained_model(dataset_id: str, metadata: Dict) -> Dict:
    """Register a newly trained model"""
    return ModelRegistry.register_model(dataset_id, metadata)

def get_active_model_metadata(dataset_id: str) -> Optional[Dict]:
    """Get metadata for the active model version"""
    active_version = ModelRegistry.get_active_version(dataset_id)
    return active_version["metadata"] if active_version else None

def get_model_history(dataset_id: str) -> List[Dict]:
    """Get training history for a dataset"""
    versions = ModelRegistry.get_model_versions(dataset_id)
    return [{
        "version_id": v["version_id"],
        "created_at": v["created_at"],
        "best_model": v["metadata"].get("best_model"),
        "best_score": v["metadata"].get("best_score"),
        "task": v["metadata"].get("task"),
        "training_time_seconds": v["metadata"].get("training_time_seconds"),
        "is_active": v["version_id"] == ModelRegistry.load_registry()["models"].get(dataset_id, {}).get("active_version")
    } for v in versions]

def rollback_model(dataset_id: str, version_id: str) -> Dict:
    """Rollback to a previous model version"""
    return ModelRegistry.set_active_version(dataset_id, version_id)