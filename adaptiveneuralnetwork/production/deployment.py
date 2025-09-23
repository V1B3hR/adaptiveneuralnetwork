"""
Kubernetes-based deployment and auto-scaling for adaptive neural networks.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DeploymentConfig:
    """Configuration for Kubernetes deployment."""
    name: str
    namespace: str = "default"
    replicas: int = 3
    image: str = "adaptiveneuralnetwork:latest"
    cpu_request: str = "500m" 
    cpu_limit: str = "2"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    gpu_request: int = 0
    gpu_limit: int = 1
    port: int = 8080
    enable_autoscaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80


class KubernetesDeployment:
    """Kubernetes deployment manager for adaptive neural networks."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifests_dir = Path("k8s")
        self.manifests_dir.mkdir(exist_ok=True)
    
    def generate_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment", 
            "metadata": {
                "name": self.config.name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.name,
                    "component": "adaptive-neural-network"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.name
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "adaptive-nn",
                            "image": self.config.image,
                            "ports": [{
                                "containerPort": self.config.port,
                                "name": "http"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": self.config.cpu_request,
                                    "memory": self.config.memory_request
                                },
                                "limits": {
                                    "cpu": self.config.cpu_limit,
                                    "memory": self.config.memory_limit
                                }
                            },
                            "env": [
                                {
                                    "name": "MODEL_PATH",
                                    "value": "/models"
                                },
                                {
                                    "name": "PORT",
                                    "value": str(self.config.port)
                                }
                            ],
                            "volumeMounts": [{
                                "name": "model-storage",
                                "mountPath": "/models"
                            }],
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.config.port
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": self.config.port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }],
                        "volumes": [{
                            "name": "model-storage",
                            "persistentVolumeClaim": {
                                "claimName": f"{self.config.name}-models"
                            }
                        }]
                    }
                }
            }
        }
        
        # Add GPU resources if requested
        if self.config.gpu_request > 0:
            resources = manifest["spec"]["template"]["spec"]["containers"][0]["resources"]
            resources["requests"]["nvidia.com/gpu"] = str(self.config.gpu_request)
            resources["limits"]["nvidia.com/gpu"] = str(self.config.gpu_limit)
        
        return manifest
    
    def generate_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.name}-service",
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.name
                }
            },
            "spec": {
                "selector": {
                    "app": self.config.name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": self.config.port,
                    "protocol": "TCP",
                    "name": "http"
                }],
                "type": "ClusterIP"
            }
        }
    
    def generate_hpa_manifest(self) -> Dict[str, Any]:
        """Generate Horizontal Pod Autoscaler manifest."""
        if not self.config.enable_autoscaling:
            return {}
        
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.name}-hpa",
                "namespace": self.config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.config.name
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas, 
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization", 
                                "averageUtilization": self.config.target_cpu_utilization
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.config.target_memory_utilization
                            }
                        }
                    }
                ]
            }
        }
    
    def generate_pvc_manifest(self) -> Dict[str, Any]:
        """Generate PersistentVolumeClaim manifest for model storage."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config.name}-models",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {
                    "requests": {
                        "storage": "10Gi"
                    }
                },
                "storageClassName": "default"
            }
        }
    
    def write_manifests(self) -> List[Path]:
        """Write all Kubernetes manifests to files."""
        manifests = [
            ("deployment.yaml", self.generate_deployment_manifest()),
            ("service.yaml", self.generate_service_manifest()),
            ("pvc.yaml", self.generate_pvc_manifest())
        ]
        
        if self.config.enable_autoscaling:
            manifests.append(("hpa.yaml", self.generate_hpa_manifest()))
        
        written_files = []
        for filename, manifest in manifests:
            if not manifest:  # Skip empty manifests
                continue
                
            file_path = self.manifests_dir / filename
            with open(file_path, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False)
            written_files.append(file_path)
        
        return written_files


class AutoScaler:
    """Auto-scaling based on cognitive load metrics."""
    
    def __init__(self, deployment_name: str, namespace: str = "default"):
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.metrics_history = []
    
    def calculate_cognitive_load(self, model_metrics: Dict[str, float]) -> float:
        """Calculate cognitive load from model metrics."""
        # Factors that contribute to cognitive load:
        # - Processing latency
        # - Memory usage
        # - Active node ratio
        # - Trust network complexity
        # - Energy distribution variance
        
        latency_factor = min(model_metrics.get("avg_latency_ms", 0) / 100.0, 1.0)
        memory_factor = model_metrics.get("memory_usage_percent", 0) / 100.0
        active_nodes_factor = model_metrics.get("active_node_ratio", 0.5)
        trust_complexity = model_metrics.get("trust_network_complexity", 0.5)
        energy_variance = model_metrics.get("energy_distribution_variance", 0.3)
        
        # Weighted cognitive load calculation
        cognitive_load = (
            0.3 * latency_factor +
            0.2 * memory_factor +
            0.2 * active_nodes_factor +
            0.15 * trust_complexity +
            0.15 * energy_variance
        )
        
        return min(cognitive_load, 1.0)
    
    def should_scale_up(self, cognitive_load: float, threshold: float = 0.8) -> bool:
        """Determine if scaling up is needed based on cognitive load."""
        self.metrics_history.append(cognitive_load)
        
        # Keep only last 10 measurements
        if len(self.metrics_history) > 10:
            self.metrics_history = self.metrics_history[-10:]
        
        # Scale up if average load over last 5 measurements exceeds threshold
        if len(self.metrics_history) >= 5:
            recent_avg = sum(self.metrics_history[-5:]) / 5
            return recent_avg > threshold
        
        return cognitive_load > threshold
    
    def should_scale_down(self, cognitive_load: float, threshold: float = 0.3) -> bool:
        """Determine if scaling down is possible based on cognitive load."""
        if len(self.metrics_history) >= 10:
            recent_avg = sum(self.metrics_history[-10:]) / 10
            return recent_avg < threshold
        
        return cognitive_load < threshold
    
    def get_scaling_recommendation(self, current_replicas: int, cognitive_load: float) -> int:
        """Get scaling recommendation based on cognitive load."""
        if self.should_scale_up(cognitive_load):
            # Scale up by 50% or at least 1 replica
            return max(current_replicas + 1, int(current_replicas * 1.5))
        elif self.should_scale_down(cognitive_load):
            # Scale down by 25% but keep at least 1 replica
            return max(1, int(current_replicas * 0.75))
        
        return current_replicas