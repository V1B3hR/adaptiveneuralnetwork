"""
Advanced Analytics Dashboard - Phase 4.1

This module provides enhanced analytics capabilities including:
- Real-time network topology visualization
- Performance degradation early warning systems  
- Trust network flow analysis
- Energy distribution heat maps
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NetworkTopologyNode:
    """Represents a node in the network topology"""
    node_id: str
    node_type: str  # agent, decision_hub, memory_bank, etc.
    position: tuple[float, float, float]  # 3D coordinates
    connections: list[str]  # Connected node IDs
    activity_level: float  # 0.0 to 1.0
    trust_score: float  # 0.0 to 1.0
    energy_consumption: float
    last_activity: float  # timestamp
    metadata: dict[str, Any]


@dataclass
class NetworkTopologyEdge:
    """Represents an edge in the network topology"""
    edge_id: str
    source_node: str
    target_node: str
    connection_type: str  # trust, communication, dependency, etc.
    strength: float  # 0.0 to 1.0
    data_flow_rate: float  # units per second
    latency: float  # milliseconds
    last_active: float  # timestamp
    metadata: dict[str, Any]


@dataclass
class PerformanceMetric:
    """Performance metric with historical tracking"""
    metric_name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    history: list[tuple[float, float]]  # (timestamp, value)
    trend: str  # "improving", "stable", "degrading"
    last_updated: float


@dataclass
class EnergyDistribution:
    """Energy distribution data for heat map visualization"""
    node_id: str
    energy_consumption: float
    energy_efficiency: float  # output/energy ratio
    heat_level: float  # 0.0 to 1.0 for visualization
    cooling_rate: float
    optimal_range: tuple[float, float]
    timestamp: float


class AdvancedAnalyticsDashboard:
    """Enhanced analytics dashboard for Phase 4.1"""

    def __init__(self, max_history_points: int = 1000):
        self.max_history_points = max_history_points

        # Network topology tracking
        self.network_nodes: dict[str, NetworkTopologyNode] = {}
        self.network_edges: dict[str, NetworkTopologyEdge] = {}
        self.topology_history: deque = deque(maxlen=100)  # Last 100 topology snapshots

        # Performance monitoring
        self.performance_metrics: dict[str, PerformanceMetric] = {}
        self.alerts: list[dict[str, Any]] = []
        self.alert_history: deque = deque(maxlen=500)

        # Trust network analysis
        self.trust_flows: dict[str, list[tuple[str, str, float, float]]] = defaultdict(list)  # timestamp -> [(source, target, trust, flow)]
        self.trust_clusters: list[set[str]] = []
        self.trust_anomalies: list[dict[str, Any]] = []

        # Energy tracking
        self.energy_distributions: dict[str, EnergyDistribution] = {}
        self.energy_history: deque = deque(maxlen=max_history_points)
        self.energy_alerts: list[dict[str, Any]] = []

        logger.info("Advanced Analytics Dashboard initialized")

    def add_network_node(self, node_id: str, node_type: str, position: tuple[float, float, float],
                        connections: list[str] = None, activity_level: float = 0.0,
                        trust_score: float = 0.5, energy_consumption: float = 0.0,
                        metadata: dict[str, Any] = None) -> None:
        """Add or update a network topology node"""

        node = NetworkTopologyNode(
            node_id=node_id,
            node_type=node_type,
            position=position,
            connections=connections or [],
            activity_level=activity_level,
            trust_score=trust_score,
            energy_consumption=energy_consumption,
            last_activity=time.time(),
            metadata=metadata or {}
        )

        self.network_nodes[node_id] = node
        logger.debug(f"Added/updated network node: {node_id}")

    def add_network_edge(self, edge_id: str, source_node: str, target_node: str,
                        connection_type: str, strength: float = 1.0,
                        data_flow_rate: float = 0.0, latency: float = 0.0,
                        metadata: dict[str, Any] = None) -> None:
        """Add or update a network topology edge"""

        edge = NetworkTopologyEdge(
            edge_id=edge_id,
            source_node=source_node,
            target_node=target_node,
            connection_type=connection_type,
            strength=strength,
            data_flow_rate=data_flow_rate,
            latency=latency,
            last_active=time.time(),
            metadata=metadata or {}
        )

        self.network_edges[edge_id] = edge

        # Update node connections
        if source_node in self.network_nodes and target_node not in self.network_nodes[source_node].connections:
            self.network_nodes[source_node].connections.append(target_node)
        if target_node in self.network_nodes and source_node not in self.network_nodes[target_node].connections:
            self.network_nodes[target_node].connections.append(source_node)

        logger.debug(f"Added/updated network edge: {edge_id}")

    def get_real_time_topology_visualization(self) -> dict[str, Any]:
        """Generate real-time network topology visualization data"""

        # Create nodes array for visualization
        nodes = []
        for node_id, node in self.network_nodes.items():
            viz_node = {
                "id": node_id,
                "label": f"{node.node_type}: {node_id}",
                "type": node.node_type,
                "position": {
                    "x": node.position[0],
                    "y": node.position[1],
                    "z": node.position[2]
                },
                "size": max(10, node.activity_level * 50),  # Scale size by activity
                "color": self._get_node_color(node.trust_score, node.activity_level),
                "activity_level": node.activity_level,
                "trust_score": node.trust_score,
                "energy_consumption": node.energy_consumption,
                "last_activity": node.last_activity,
                "metadata": node.metadata
            }
            nodes.append(viz_node)

        # Create edges array for visualization
        edges = []
        for edge_id, edge in self.network_edges.items():
            viz_edge = {
                "id": edge_id,
                "source": edge.source_node,
                "target": edge.target_node,
                "type": edge.connection_type,
                "strength": edge.strength,
                "width": max(1, edge.data_flow_rate * 10),  # Scale width by flow rate
                "color": self._get_edge_color(edge.connection_type, edge.strength),
                "data_flow_rate": edge.data_flow_rate,
                "latency": edge.latency,
                "last_active": edge.last_active,
                "animated": edge.data_flow_rate > 0.1,  # Animate high-flow edges
                "metadata": edge.metadata
            }
            edges.append(viz_edge)

        # Calculate network statistics
        stats = self._calculate_network_statistics()

        visualization_data = {
            "timestamp": time.time(),
            "nodes": nodes,
            "edges": edges,
            "statistics": stats,
            "layout_algorithm": "force_directed_3d",
            "interaction_options": {
                "zoom": True,
                "pan": True,
                "node_selection": True,
                "edge_selection": True,
                "clustering": True
            }
        }

        # Store snapshot for history
        self.topology_history.append({
            "timestamp": time.time(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "avg_activity": stats.get("average_activity", 0),
            "avg_trust": stats.get("average_trust", 0)
        })

        return visualization_data

    def add_performance_metric(self, metric_name: str, current_value: float,
                             threshold_warning: float, threshold_critical: float) -> None:
        """Add or update a performance metric"""

        current_time = time.time()

        if metric_name in self.performance_metrics:
            metric = self.performance_metrics[metric_name]
            metric.history.append((current_time, current_value))

            # Keep only recent history
            if len(metric.history) > self.max_history_points:
                metric.history = metric.history[-self.max_history_points:]

            # Calculate trend
            metric.trend = self._calculate_trend(metric.history)
            metric.current_value = current_value
            metric.last_updated = current_time
        else:
            metric = PerformanceMetric(
                metric_name=metric_name,
                current_value=current_value,
                threshold_warning=threshold_warning,
                threshold_critical=threshold_critical,
                history=[(current_time, current_value)],
                trend="stable",
                last_updated=current_time
            )
            self.performance_metrics[metric_name] = metric

        # Check for alerts
        self._check_performance_alerts(metric)

    def get_performance_degradation_warnings(self) -> list[dict[str, Any]]:
        """Get early warning system alerts for performance degradation"""

        warnings = []
        current_time = time.time()

        for metric_name, metric in self.performance_metrics.items():
            warning_level = "none"

            # Check threshold violations
            if metric.current_value >= metric.threshold_critical:
                warning_level = "critical"
            elif metric.current_value >= metric.threshold_warning:
                warning_level = "warning"

            # Check trend degradation
            trend_severity = "none"
            if metric.trend == "degrading" and len(metric.history) >= 5:
                recent_values = [v for _, v in metric.history[-5:]]
                degradation_rate = (recent_values[-1] - recent_values[0]) / len(recent_values)

                if degradation_rate > metric.threshold_warning * 0.1:  # 10% of warning threshold per measurement
                    trend_severity = "significant"
                elif degradation_rate > 0:
                    trend_severity = "moderate"

            if warning_level != "none" or trend_severity != "none":
                warning = {
                    "metric_name": metric_name,
                    "current_value": metric.current_value,
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                    "warning_level": warning_level,
                    "trend": metric.trend,
                    "trend_severity": trend_severity,
                    "time_since_last_update": current_time - metric.last_updated,
                    "prediction": self._predict_metric_trajectory(metric),
                    "recommended_actions": self._get_recommended_actions(metric_name, warning_level, trend_severity)
                }
                warnings.append(warning)

        return warnings

    def analyze_trust_network_flows(self) -> dict[str, Any]:
        """Analyze trust network flows and detect patterns"""

        current_time = time.time()

        # Collect trust relationships from network topology
        trust_relationships = []
        for edge_id, edge in self.network_edges.items():
            if edge.connection_type == "trust":
                trust_relationships.append({
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "trust_level": edge.strength,
                    "flow_rate": edge.data_flow_rate,
                    "timestamp": edge.last_active
                })

        # Detect trust clusters using connected components
        trust_clusters = self._detect_trust_clusters(trust_relationships)

        # Calculate trust flow metrics
        flow_metrics = {
            "total_trust_connections": len(trust_relationships),
            "average_trust_level": np.mean([r["trust_level"] for r in trust_relationships]) if trust_relationships else 0,
            "trust_variance": np.var([r["trust_level"] for r in trust_relationships]) if trust_relationships else 0,
            "high_trust_connections": len([r for r in trust_relationships if r["trust_level"] > 0.8]),
            "low_trust_connections": len([r for r in trust_relationships if r["trust_level"] < 0.3]),
            "active_flows": len([r for r in trust_relationships if r["flow_rate"] > 0.1])
        }

        # Detect anomalies
        anomalies = self._detect_trust_anomalies(trust_relationships)

        # Generate trust flow visualization data
        flow_visualization = {
            "nodes": [
                {
                    "id": node_id,
                    "trust_centrality": self._calculate_trust_centrality(node_id, trust_relationships),
                    "in_degree": len([r for r in trust_relationships if r["target"] == node_id]),
                    "out_degree": len([r for r in trust_relationships if r["source"] == node_id])
                }
                for node_id in self.network_nodes.keys()
            ],
            "flows": [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "trust_level": r["trust_level"],
                    "flow_rate": r["flow_rate"],
                    "flow_direction": "bidirectional" if any(
                        other["source"] == r["target"] and other["target"] == r["source"]
                        for other in trust_relationships
                    ) else "unidirectional"
                }
                for r in trust_relationships
            ]
        }

        analysis_result = {
            "timestamp": current_time,
            "trust_clusters": trust_clusters,
            "flow_metrics": flow_metrics,
            "anomalies": anomalies,
            "visualization": flow_visualization,
            "recommendations": self._generate_trust_recommendations(trust_clusters, flow_metrics, anomalies)
        }

        return analysis_result

    def update_energy_distribution(self, node_id: str, energy_consumption: float,
                                 energy_efficiency: float, optimal_range: tuple[float, float] = None) -> None:
        """Update energy distribution data for a node"""

        current_time = time.time()

        # Calculate heat level (0.0 to 1.0)
        if optimal_range is None:
            optimal_range = (0.0, 1.0)

        optimal_min, optimal_max = optimal_range
        if energy_consumption <= optimal_max:
            heat_level = max(0.0, (energy_consumption - optimal_min) / (optimal_max - optimal_min))
        else:
            # Exponential increase for values above optimal
            excess = energy_consumption - optimal_max
            heat_level = min(1.0, 0.8 + 0.2 * (1 - np.exp(-excess)))

        # Calculate cooling rate based on efficiency
        cooling_rate = energy_efficiency * 0.1  # Higher efficiency = better cooling

        energy_dist = EnergyDistribution(
            node_id=node_id,
            energy_consumption=energy_consumption,
            energy_efficiency=energy_efficiency,
            heat_level=heat_level,
            cooling_rate=cooling_rate,
            optimal_range=optimal_range,
            timestamp=current_time
        )

        self.energy_distributions[node_id] = energy_dist

        # Update network node energy if it exists
        if node_id in self.network_nodes:
            self.network_nodes[node_id].energy_consumption = energy_consumption

        # Store in history
        self.energy_history.append({
            "timestamp": current_time,
            "node_id": node_id,
            "energy_consumption": energy_consumption,
            "energy_efficiency": energy_efficiency,
            "heat_level": heat_level
        })

        # Check for energy alerts
        self._check_energy_alerts(energy_dist)

    def get_energy_distribution_heatmap(self) -> dict[str, Any]:
        """Generate energy distribution heat map data"""

        current_time = time.time()

        # Prepare heat map data
        heatmap_data = []
        for node_id, energy_dist in self.energy_distributions.items():
            if node_id in self.network_nodes:
                node = self.network_nodes[node_id]
                heatmap_point = {
                    "node_id": node_id,
                    "position": {
                        "x": node.position[0],
                        "y": node.position[1],
                        "z": node.position[2]
                    },
                    "heat_level": energy_dist.heat_level,
                    "energy_consumption": energy_dist.energy_consumption,
                    "energy_efficiency": energy_dist.energy_efficiency,
                    "cooling_rate": energy_dist.cooling_rate,
                    "optimal_range": energy_dist.optimal_range,
                    "status": self._get_energy_status(energy_dist),
                    "last_updated": energy_dist.timestamp
                }
                heatmap_data.append(heatmap_point)

        # Calculate statistics
        if heatmap_data:
            heat_levels = [point["heat_level"] for point in heatmap_data]
            energy_consumptions = [point["energy_consumption"] for point in heatmap_data]

            statistics = {
                "total_nodes": len(heatmap_data),
                "average_heat_level": np.mean(heat_levels),
                "max_heat_level": np.max(heat_levels),
                "average_energy_consumption": np.mean(energy_consumptions),
                "total_energy_consumption": np.sum(energy_consumptions),
                "overheating_nodes": len([p for p in heatmap_data if p["heat_level"] > 0.8]),
                "efficient_nodes": len([p for p in heatmap_data if p["energy_efficiency"] > 0.7])
            }
        else:
            statistics = {"total_nodes": 0}

        heatmap_result = {
            "timestamp": current_time,
            "heatmap_data": heatmap_data,
            "statistics": statistics,
            "color_scale": {
                "min_color": "#0000FF",  # Blue for cool
                "mid_color": "#FFFF00",  # Yellow for warm
                "max_color": "#FF0000"   # Red for hot
            },
            "visualization_options": {
                "interpolation": "linear",
                "opacity": 0.7,
                "show_grid": True,
                "show_contours": True
            },
            "alerts": [alert for alert in self.energy_alerts if current_time - alert["timestamp"] < 300]  # Last 5 minutes
        }

        return heatmap_result

    def get_comprehensive_dashboard(self) -> dict[str, Any]:
        """Get comprehensive dashboard combining all analytics"""

        current_time = time.time()

        dashboard = {
            "timestamp": current_time,
            "network_topology": self.get_real_time_topology_visualization(),
            "performance_warnings": self.get_performance_degradation_warnings(),
            "trust_analysis": self.analyze_trust_network_flows(),
            "energy_heatmap": self.get_energy_distribution_heatmap(),
            "system_health": {
                "overall_status": self._calculate_overall_health(),
                "active_alerts": len(self.alerts),
                "nodes_monitored": len(self.network_nodes),
                "edges_monitored": len(self.network_edges),
                "metrics_tracked": len(self.performance_metrics),
                "last_updated": current_time
            }
        }

        return dashboard

    # Helper methods
    def _get_node_color(self, trust_score: float, activity_level: float) -> str:
        """Generate color for node based on trust and activity"""
        # Blend trust (hue) and activity (saturation)
        hue = int(trust_score * 120)  # 0-120 degrees (red to green)
        saturation = int(activity_level * 100)  # 0-100%
        return f"hsl({hue}, {saturation}%, 50%)"

    def _get_edge_color(self, connection_type: str, strength: float) -> str:
        """Generate color for edge based on type and strength"""
        type_colors = {
            "trust": (0, 255, 0),     # Green
            "communication": (0, 0, 255),  # Blue
            "dependency": (255, 165, 0),   # Orange
            "conflict": (255, 0, 0)        # Red
        }

        base_color = type_colors.get(connection_type, (128, 128, 128))
        alpha = strength  # Use strength as alpha

        return f"rgba({base_color[0]}, {base_color[1]}, {base_color[2]}, {alpha})"

    def _calculate_network_statistics(self) -> dict[str, Any]:
        """Calculate network topology statistics"""
        if not self.network_nodes:
            return {}

        activities = [node.activity_level for node in self.network_nodes.values()]
        trust_scores = [node.trust_score for node in self.network_nodes.values()]
        energies = [node.energy_consumption for node in self.network_nodes.values()]

        return {
            "node_count": len(self.network_nodes),
            "edge_count": len(self.network_edges),
            "average_activity": np.mean(activities) if activities else 0,
            "average_trust": np.mean(trust_scores) if trust_scores else 0,
            "total_energy": np.sum(energies) if energies else 0,
            "network_density": len(self.network_edges) / max(1, len(self.network_nodes) * (len(self.network_nodes) - 1) / 2),
            "connected_components": self._count_connected_components()
        }

    def _calculate_trend(self, history: list[tuple[float, float]]) -> str:
        """Calculate trend from historical data"""
        if len(history) < 3:
            return "stable"

        recent_values = [v for _, v in history[-5:]]
        if len(recent_values) < 2:
            return "stable"

        # Simple linear regression slope
        n = len(recent_values)
        x = list(range(n))
        slope = (n * sum(i * v for i, v in enumerate(recent_values)) - sum(x) * sum(recent_values)) / (n * sum(i*i for i in x) - sum(x)**2)

        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

    def _check_performance_alerts(self, metric: PerformanceMetric) -> None:
        """Check and generate performance alerts"""
        current_time = time.time()

        if metric.current_value >= metric.threshold_critical:
            self._add_alert("critical", f"Critical threshold exceeded for {metric.metric_name}",
                          {"metric": metric.metric_name, "value": metric.current_value, "threshold": metric.threshold_critical})
        elif metric.current_value >= metric.threshold_warning:
            self._add_alert("warning", f"Warning threshold exceeded for {metric.metric_name}",
                          {"metric": metric.metric_name, "value": metric.current_value, "threshold": metric.threshold_warning})

    def _predict_metric_trajectory(self, metric: PerformanceMetric) -> dict[str, Any]:
        """Predict future trajectory of a metric"""
        if len(metric.history) < 5:
            return {"prediction": "insufficient_data"}

        recent_values = [v for _, v in metric.history[-10:]]

        # Simple linear extrapolation
        if len(recent_values) >= 2:
            slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
            predicted_value = recent_values[-1] + slope * 5  # Predict 5 steps ahead

            confidence = max(0.1, min(0.9, 1.0 - abs(slope) / max(recent_values)))

            return {
                "prediction": "linear_extrapolation",
                "predicted_value": predicted_value,
                "confidence": confidence,
                "time_to_warning": self._calculate_time_to_threshold(metric, metric.threshold_warning, slope),
                "time_to_critical": self._calculate_time_to_threshold(metric, metric.threshold_critical, slope)
            }

        return {"prediction": "stable", "predicted_value": metric.current_value}

    def _get_recommended_actions(self, metric_name: str, warning_level: str, trend_severity: str) -> list[str]:
        """Get recommended actions for performance issues"""
        actions = []

        if warning_level == "critical":
            actions.append(f"Immediate intervention required for {metric_name}")
            actions.append("Consider emergency protocols")
        elif warning_level == "warning":
            actions.append(f"Monitor {metric_name} closely")
            actions.append("Prepare contingency measures")

        if trend_severity == "significant":
            actions.append("Investigate root cause of degradation")
            actions.append("Consider system optimization")

        return actions

    def _detect_trust_clusters(self, trust_relationships: list[dict[str, Any]]) -> list[set[str]]:
        """Detect trust clusters using connected components"""
        # Build adjacency list for high-trust connections
        graph = defaultdict(set)
        for rel in trust_relationships:
            if rel["trust_level"] > 0.6:  # Only consider high-trust connections
                graph[rel["source"]].add(rel["target"])
                graph[rel["target"]].add(rel["source"])

        # Find connected components
        visited = set()
        clusters = []

        def dfs(node, cluster):
            if node in visited:
                return
            visited.add(node)
            cluster.add(node)
            for neighbor in graph[node]:
                dfs(neighbor, cluster)

        for node in graph:
            if node not in visited:
                cluster = set()
                dfs(node, cluster)
                if len(cluster) > 1:  # Only keep clusters with multiple nodes
                    clusters.append(cluster)

        return clusters

    def _detect_trust_anomalies(self, trust_relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect trust anomalies"""
        anomalies = []

        if not trust_relationships:
            return anomalies

        trust_levels = [r["trust_level"] for r in trust_relationships]
        mean_trust = np.mean(trust_levels)
        std_trust = np.std(trust_levels)

        # Detect outliers (trust levels > 2 standard deviations from mean)
        for rel in trust_relationships:
            z_score = abs(rel["trust_level"] - mean_trust) / max(std_trust, 0.1)
            if z_score > 2:
                anomalies.append({
                    "type": "trust_outlier",
                    "source": rel["source"],
                    "target": rel["target"],
                    "trust_level": rel["trust_level"],
                    "z_score": z_score,
                    "severity": "high" if z_score > 3 else "medium"
                })

        return anomalies

    def _calculate_trust_centrality(self, node_id: str, trust_relationships: list[dict[str, Any]]) -> float:
        """Calculate trust centrality for a node"""
        incoming_trust = sum(r["trust_level"] for r in trust_relationships if r["target"] == node_id)
        outgoing_trust = sum(r["trust_level"] for r in trust_relationships if r["source"] == node_id)

        total_possible = len(self.network_nodes) - 1  # Exclude self
        return (incoming_trust + outgoing_trust) / max(1, total_possible * 2)

    def _generate_trust_recommendations(self, clusters: list[set[str]],
                                      metrics: dict[str, Any], anomalies: list[dict[str, Any]]) -> list[str]:
        """Generate trust network recommendations"""
        recommendations = []

        if metrics["average_trust_level"] < 0.5:
            recommendations.append("Overall trust levels are low - consider trust-building initiatives")

        if len(clusters) > len(self.network_nodes) / 3:
            recommendations.append("Network is highly fragmented - consider bridge-building between clusters")

        if metrics["trust_variance"] > 0.3:
            recommendations.append("High trust variance detected - investigate inconsistent trust patterns")

        if len(anomalies) > 0:
            recommendations.append(f"Found {len(anomalies)} trust anomalies requiring investigation")

        return recommendations

    def _check_energy_alerts(self, energy_dist: EnergyDistribution) -> None:
        """Check and generate energy alerts"""
        current_time = time.time()

        if energy_dist.heat_level > 0.9:
            self._add_energy_alert("critical", f"Node {energy_dist.node_id} is overheating",
                                 {"node_id": energy_dist.node_id, "heat_level": energy_dist.heat_level})
        elif energy_dist.heat_level > 0.7:
            self._add_energy_alert("warning", f"Node {energy_dist.node_id} is running hot",
                                 {"node_id": energy_dist.node_id, "heat_level": energy_dist.heat_level})

        if energy_dist.energy_efficiency < 0.3:
            self._add_energy_alert("warning", f"Node {energy_dist.node_id} has low energy efficiency",
                                 {"node_id": energy_dist.node_id, "efficiency": energy_dist.energy_efficiency})

    def _get_energy_status(self, energy_dist: EnergyDistribution) -> str:
        """Get energy status for a node"""
        if energy_dist.heat_level > 0.9:
            return "critical"
        elif energy_dist.heat_level > 0.7:
            return "warning"
        elif energy_dist.energy_efficiency < 0.3:
            return "inefficient"
        else:
            return "normal"

    def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        critical_alerts = len([a for a in self.alerts if a["level"] == "critical"])
        warning_alerts = len([a for a in self.alerts if a["level"] == "warning"])

        if critical_alerts > 0:
            return "critical"
        elif warning_alerts > 5:
            return "degraded"
        elif warning_alerts > 0:
            return "warning"
        else:
            return "healthy"

    def _count_connected_components(self) -> int:
        """Count connected components in the network"""
        if not self.network_nodes:
            return 0

        visited = set()
        components = 0

        def dfs(node_id):
            if node_id in visited or node_id not in self.network_nodes:
                return
            visited.add(node_id)
            for connected_id in self.network_nodes[node_id].connections:
                dfs(connected_id)

        for node_id in self.network_nodes:
            if node_id not in visited:
                dfs(node_id)
                components += 1

        return components

    def _calculate_time_to_threshold(self, metric: PerformanceMetric, threshold: float, slope: float) -> float | None:
        """Calculate time until metric reaches threshold"""
        if slope <= 0:
            return None  # Won't reach threshold with current trend

        time_steps = (threshold - metric.current_value) / slope
        return max(0, time_steps) if time_steps > 0 else None

    def _add_alert(self, level: str, message: str, details: dict[str, Any]) -> None:
        """Add a general alert"""
        alert = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "details": details,
            "id": f"alert_{len(self.alerts)}"
        }

        self.alerts.append(alert)
        self.alert_history.append(alert)

        logger.warning(f"Alert [{level}]: {message}")

    def _add_energy_alert(self, level: str, message: str, details: dict[str, Any]) -> None:
        """Add an energy-specific alert"""
        alert = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "details": details,
            "type": "energy",
            "id": f"energy_alert_{len(self.energy_alerts)}"
        }

        self.energy_alerts.append(alert)
        logger.warning(f"Energy Alert [{level}]: {message}")
