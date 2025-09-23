"""
Decision Transparency System - Phase 4.1

This module provides:
- Attention mechanisms for decision tracking
- Counterfactual reasoning capabilities
- Natural language explanations for network decisions
- Enhanced audit trail systems for ethical compliance
"""

import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple, Callable
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class AttentionWeight:
    """Represents attention weight for a specific input or decision factor"""
    factor_id: str
    factor_name: str
    weight: float  # 0.0 to 1.0
    context: Dict[str, Any]
    timestamp: float
    layer_id: str  # Which decision layer this applies to


@dataclass
class AttentionMap:
    """Attention map for a decision showing what factors were attended to"""
    decision_id: str
    attention_weights: List[AttentionWeight]
    focus_regions: List[Tuple[str, float]]  # (region_name, intensity)
    temporal_sequence: List[Tuple[float, str, float]]  # (timestamp, factor_id, weight)
    overall_confidence: float
    attention_entropy: float  # Measure of attention dispersion


@dataclass
class CounterfactualScenario:
    """Represents a counterfactual scenario for 'what if' analysis"""
    scenario_id: str
    original_decision_id: str
    changed_factors: Dict[str, Any]
    predicted_outcome: Any
    confidence: float
    probability: float
    explanation: str
    impact_assessment: Dict[str, float]
    timestamp: float


@dataclass
class NaturalLanguageExplanation:
    """Natural language explanation of a decision"""
    decision_id: str
    explanation_type: str  # "summary", "detailed", "technical", "ethical"
    explanation_text: str
    key_factors: List[str]
    confidence_phrases: List[str]
    reasoning_chain_text: str
    ethical_considerations_text: str
    alternative_explanations: List[str]
    readability_score: float
    timestamp: float


class AttentionMechanism(ABC):
    """Abstract base class for attention mechanisms"""
    
    @abstractmethod
    def calculate_attention(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> AttentionMap:
        """Calculate attention weights for given inputs and context"""
        pass
    
    @abstractmethod
    def visualize_attention(self, attention_map: AttentionMap) -> Dict[str, Any]:
        """Generate visualization data for attention patterns"""
        pass


class MultiHeadAttentionTracker(AttentionMechanism):
    """Multi-head attention mechanism for decision tracking"""
    
    def __init__(self, num_heads: int = 8, head_dim: int = 64):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention_history: Dict[str, List[AttentionMap]] = defaultdict(list)
        
    def calculate_attention(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> AttentionMap:
        """Calculate multi-head attention weights"""
        
        decision_id = context.get("decision_id", f"decision_{time.time()}")
        current_time = time.time()
        
        # Extract input features
        input_features = self._extract_features(inputs)
        context_features = self._extract_features(context)
        
        # Calculate attention for each head
        attention_weights = []
        focus_regions = []
        temporal_sequence = []
        
        for head_idx in range(self.num_heads):
            head_weights = self._calculate_head_attention(input_features, context_features, head_idx)
            
            for factor_id, weight in head_weights.items():
                if weight > 0.1:  # Only keep significant weights
                    attention_weight = AttentionWeight(
                        factor_id=factor_id,
                        factor_name=self._get_factor_name(factor_id, inputs),
                        weight=weight,
                        context={"head": head_idx, "layer": context.get("layer", "default")},
                        timestamp=current_time,
                        layer_id=context.get("layer", "default")
                    )
                    attention_weights.append(attention_weight)
                    temporal_sequence.append((current_time, factor_id, weight))
        
        # Calculate focus regions (grouping similar factors)
        focus_regions = self._calculate_focus_regions(attention_weights)
        
        # Calculate overall confidence and entropy
        weights_array = np.array([w.weight for w in attention_weights])
        overall_confidence = np.max(weights_array) if len(weights_array) > 0 else 0.0
        attention_entropy = self._calculate_entropy(weights_array)
        
        attention_map = AttentionMap(
            decision_id=decision_id,
            attention_weights=attention_weights,
            focus_regions=focus_regions,
            temporal_sequence=temporal_sequence,
            overall_confidence=overall_confidence,
            attention_entropy=attention_entropy
        )
        
        # Store in history
        self.attention_history[decision_id].append(attention_map)
        
        return attention_map
    
    def visualize_attention(self, attention_map: AttentionMap) -> Dict[str, Any]:
        """Generate attention visualization data"""
        
        # Prepare data for heatmap visualization
        heatmap_data = []
        for weight in attention_map.attention_weights:
            heatmap_data.append({
                "factor": weight.factor_name,
                "weight": weight.weight,
                "layer": weight.layer_id,
                "timestamp": weight.timestamp
            })
        
        # Prepare temporal flow data
        temporal_data = [
            {
                "time": seq[0],
                "factor": seq[1],
                "weight": seq[2],
                "position": idx
            }
            for idx, seq in enumerate(attention_map.temporal_sequence)
        ]
        
        # Prepare focus region data
        focus_data = [
            {
                "region": region[0],
                "intensity": region[1],
                "size": len([w for w in attention_map.attention_weights if w.factor_name.startswith(region[0])])
            }
            for region in attention_map.focus_regions
        ]
        
        visualization = {
            "decision_id": attention_map.decision_id,
            "visualization_type": "multi_head_attention",
            "heatmap": {
                "data": heatmap_data,
                "color_scale": "viridis",
                "max_weight": max([w.weight for w in attention_map.attention_weights]) if attention_map.attention_weights else 1.0
            },
            "temporal_flow": {
                "data": temporal_data,
                "animation_duration": 2000,
                "show_progression": True
            },
            "focus_regions": {
                "data": focus_data,
                "visualization": "circular_plot",
                "show_connections": True
            },
            "summary": {
                "total_factors": len(attention_map.attention_weights),
                "max_attention": attention_map.overall_confidence,
                "entropy": attention_map.attention_entropy,
                "dominant_factors": [w.factor_name for w in sorted(attention_map.attention_weights, key=lambda x: x.weight, reverse=True)[:5]]
            }
        }
        
        return visualization
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from data"""
        features = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, bool):
                features[key] = 1.0 if value else 0.0
            elif isinstance(value, str):
                features[f"{key}_length"] = len(value) / 100.0  # Normalized string length
            elif isinstance(value, list):
                features[f"{key}_count"] = len(value) / 10.0  # Normalized list length
            elif isinstance(value, dict):
                features[f"{key}_size"] = len(value) / 10.0  # Normalized dict size
        
        return features
    
    def _calculate_head_attention(self, input_features: Dict[str, float], 
                                context_features: Dict[str, float], head_idx: int) -> Dict[str, float]:
        """Calculate attention weights for a specific head"""
        
        # Simulate attention calculation (in practice, this would use learned weights)
        all_features = {**input_features, **context_features}
        
        if not all_features:
            return {}
        
        # Add some randomness based on head index for diversity
        np.random.seed(head_idx * 42)
        noise = np.random.normal(0, 0.1, len(all_features))
        
        # Calculate attention weights using softmax
        feature_values = list(all_features.values())
        feature_keys = list(all_features.keys())
        
        # Add head-specific bias
        head_bias = np.sin(head_idx * np.pi / self.num_heads)
        adjusted_values = np.array(feature_values) + noise + head_bias
        
        # Apply softmax
        exp_values = np.exp(adjusted_values - np.max(adjusted_values))
        attention_weights = exp_values / np.sum(exp_values)
        
        return dict(zip(feature_keys, attention_weights))
    
    def _get_factor_name(self, factor_id: str, inputs: Dict[str, Any]) -> str:
        """Get human-readable name for a factor"""
        if factor_id in inputs:
            return factor_id
        
        # Handle derived features
        if factor_id.endswith("_length"):
            return f"Length of {factor_id[:-7]}"
        elif factor_id.endswith("_count"):
            return f"Count of {factor_id[:-6]}"
        elif factor_id.endswith("_size"):
            return f"Size of {factor_id[:-5]}"
        
        return factor_id
    
    def _calculate_focus_regions(self, attention_weights: List[AttentionWeight]) -> List[Tuple[str, float]]:
        """Calculate focus regions by grouping related factors"""
        
        # Group factors by category
        categories = defaultdict(list)
        for weight in attention_weights:
            category = self._categorize_factor(weight.factor_name)
            categories[category].append(weight.weight)
        
        # Calculate intensity for each category
        focus_regions = []
        for category, weights in categories.items():
            intensity = np.mean(weights) if weights else 0.0
            focus_regions.append((category, intensity))
        
        # Sort by intensity
        focus_regions.sort(key=lambda x: x[1], reverse=True)
        
        return focus_regions
    
    def _categorize_factor(self, factor_name: str) -> str:
        """Categorize a factor into a region"""
        factor_lower = factor_name.lower()
        
        if any(word in factor_lower for word in ["trust", "confidence", "reliability"]):
            return "Trust"
        elif any(word in factor_lower for word in ["energy", "consumption", "efficiency"]):
            return "Energy"
        elif any(word in factor_lower for word in ["time", "duration", "processing"]):
            return "Temporal"
        elif any(word in factor_lower for word in ["ethics", "compliance", "moral"]):
            return "Ethics"
        elif any(word in factor_lower for word in ["input", "data", "information"]):
            return "Input"
        else:
            return "Other"
    
    def _calculate_entropy(self, weights: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        if len(weights) == 0:
            return 0.0
        
        # Normalize weights
        normalized_weights = weights / np.sum(weights)
        
        # Calculate entropy
        entropy = -np.sum(normalized_weights * np.log2(normalized_weights + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(weights))
        
        return entropy / max(max_entropy, 1e-10)


class CounterfactualReasoner:
    """System for generating and analyzing counterfactual scenarios"""
    
    def __init__(self):
        self.scenarios: Dict[str, List[CounterfactualScenario]] = defaultdict(list)
        self.scenario_templates: Dict[str, Callable] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize counterfactual scenario templates"""
        self.scenario_templates = {
            "trust_change": self._generate_trust_change_scenario,
            "resource_variation": self._generate_resource_variation_scenario,
            "ethical_constraint": self._generate_ethical_constraint_scenario,
            "temporal_shift": self._generate_temporal_shift_scenario,
            "information_absence": self._generate_information_absence_scenario
        }
    
    def generate_counterfactual_scenarios(self, decision_id: str, original_context: Dict[str, Any],
                                        original_decision: Any, num_scenarios: int = 5) -> List[CounterfactualScenario]:
        """Generate multiple counterfactual scenarios for a decision"""
        
        scenarios = []
        current_time = time.time()
        
        # Generate scenarios using different templates
        template_names = list(self.scenario_templates.keys())
        scenarios_per_template = max(1, num_scenarios // len(template_names))
        
        for template_name in template_names:
            template_func = self.scenario_templates[template_name]
            
            for i in range(scenarios_per_template):
                scenario = template_func(
                    decision_id, original_context, original_decision, i
                )
                scenarios.append(scenario)
                
                if len(scenarios) >= num_scenarios:
                    break
            
            if len(scenarios) >= num_scenarios:
                break
        
        # Store scenarios
        self.scenarios[decision_id].extend(scenarios)
        
        return scenarios
    
    def analyze_counterfactual_impact(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Analyze the impact of counterfactual scenarios"""
        
        if not scenarios:
            return {"error": "No scenarios provided"}
        
        # Group scenarios by type
        scenario_types = defaultdict(list)
        for scenario in scenarios:
            scenario_type = scenario.explanation.split()[0]  # First word as type
            scenario_types[scenario_type].append(scenario)
        
        # Calculate impact metrics
        impact_analysis = {
            "total_scenarios": len(scenarios),
            "scenario_types": list(scenario_types.keys()),
            "average_confidence": np.mean([s.confidence for s in scenarios]),
            "average_probability": np.mean([s.probability for s in scenarios]),
            "high_impact_scenarios": [],
            "low_impact_scenarios": [],
            "confidence_distribution": self._calculate_confidence_distribution(scenarios),
            "outcome_variations": self._analyze_outcome_variations(scenarios),
            "sensitivity_analysis": self._perform_sensitivity_analysis(scenarios)
        }
        
        # Identify high and low impact scenarios
        for scenario in scenarios:
            impact_score = np.mean(list(scenario.impact_assessment.values()))
            if impact_score > 0.7:
                impact_analysis["high_impact_scenarios"].append({
                    "scenario_id": scenario.scenario_id,
                    "impact_score": impact_score,
                    "explanation": scenario.explanation
                })
            elif impact_score < 0.3:
                impact_analysis["low_impact_scenarios"].append({
                    "scenario_id": scenario.scenario_id,
                    "impact_score": impact_score,
                    "explanation": scenario.explanation
                })
        
        return impact_analysis
    
    def get_scenario_visualization(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Generate visualization data for counterfactual scenarios"""
        
        if not scenarios:
            return {"error": "No scenarios provided"}
        
        # Prepare data for different visualizations
        scenario_data = []
        for scenario in scenarios:
            scenario_data.append({
                "id": scenario.scenario_id,
                "confidence": scenario.confidence,
                "probability": scenario.probability,
                "impact_score": np.mean(list(scenario.impact_assessment.values())),
                "explanation": scenario.explanation,
                "changed_factors": list(scenario.changed_factors.keys()),
                "timestamp": scenario.timestamp
            })
        
        # Create scatter plot data (confidence vs probability)
        scatter_data = [
            {
                "x": s["confidence"],
                "y": s["probability"],
                "size": s["impact_score"] * 20,
                "label": s["explanation"][:30] + "...",
                "id": s["id"]
            }
            for s in scenario_data
        ]
        
        # Create impact distribution data
        impact_scores = [s["impact_score"] for s in scenario_data]
        impact_histogram = np.histogram(impact_scores, bins=10)
        
        histogram_data = [
            {"bin": f"{impact_histogram[1][i]:.2f}-{impact_histogram[1][i+1]:.2f}", 
             "count": int(impact_histogram[0][i])}
            for i in range(len(impact_histogram[0]))
        ]
        
        visualization = {
            "scenario_count": len(scenarios),
            "scatter_plot": {
                "data": scatter_data,
                "x_axis": "Confidence",
                "y_axis": "Probability",
                "size_axis": "Impact Score"
            },
            "impact_distribution": {
                "data": histogram_data,
                "chart_type": "histogram"
            },
            "timeline": {
                "data": [
                    {
                        "time": s["timestamp"],
                        "event": s["explanation"][:50],
                        "impact": s["impact_score"]
                    }
                    for s in sorted(scenario_data, key=lambda x: x["timestamp"])
                ]
            },
            "factor_analysis": self._analyze_changed_factors(scenarios)
        }
        
        return visualization
    
    def _generate_trust_change_scenario(self, decision_id: str, original_context: Dict[str, Any],
                                      original_decision: Any, variant: int) -> CounterfactualScenario:
        """Generate a trust change counterfactual scenario"""
        
        current_time = time.time()
        scenario_id = f"trust_change_{decision_id}_{variant}"
        
        # Modify trust levels
        trust_change = 0.2 + (variant * 0.15)  # Vary trust change
        changed_factors = {"trust_level": trust_change}
        
        # Predict outcome (simplified simulation)
        confidence = max(0.1, min(0.9, 0.7 - abs(trust_change - 0.5)))
        probability = 0.6 + (trust_change * 0.3)
        
        # Generate explanation
        explanation_text = f"Trust change scenario: If trust levels were {'increased' if trust_change > 0 else 'decreased'} by {abs(trust_change):.2f}"
        
        # Assess impact
        impact_assessment = {
            "decision_confidence": abs(trust_change) * 0.8,
            "system_stability": abs(trust_change) * 0.6,
            "ethical_compliance": abs(trust_change) * 0.4
        }
        
        return CounterfactualScenario(
            scenario_id=scenario_id,
            original_decision_id=decision_id,
            changed_factors=changed_factors,
            predicted_outcome=f"Modified decision with trust change: {trust_change}",
            confidence=confidence,
            probability=probability,
            explanation=explanation_text,
            impact_assessment=impact_assessment,
            timestamp=current_time
        )
    
    def _generate_resource_variation_scenario(self, decision_id: str, original_context: Dict[str, Any],
                                            original_decision: Any, variant: int) -> CounterfactualScenario:
        """Generate a resource variation counterfactual scenario"""
        
        current_time = time.time()
        scenario_id = f"resource_variation_{decision_id}_{variant}"
        
        # Modify resource availability
        resource_multiplier = 0.5 + (variant * 0.3)  # 0.5x to 2.0x resources
        changed_factors = {"resource_availability": resource_multiplier}
        
        # Predict outcome
        confidence = 0.8 if 0.8 <= resource_multiplier <= 1.2 else max(0.3, 0.8 - abs(resource_multiplier - 1.0))
        probability = min(0.9, 0.4 + resource_multiplier * 0.4)
        
        explanation_text = f"Resource variation scenario: With {resource_multiplier:.1f}x available resources"
        
        impact_assessment = {
            "performance": abs(resource_multiplier - 1.0) * 0.9,
            "cost_efficiency": abs(resource_multiplier - 1.0) * 0.7,
            "time_to_completion": abs(resource_multiplier - 1.0) * 0.8
        }
        
        return CounterfactualScenario(
            scenario_id=scenario_id,
            original_decision_id=decision_id,
            changed_factors=changed_factors,
            predicted_outcome=f"Resource-adjusted decision: {resource_multiplier:.1f}x resources",
            confidence=confidence,
            probability=probability,
            explanation=explanation_text,
            impact_assessment=impact_assessment,
            timestamp=current_time
        )
    
    def _generate_ethical_constraint_scenario(self, decision_id: str, original_context: Dict[str, Any],
                                            original_decision: Any, variant: int) -> CounterfactualScenario:
        """Generate an ethical constraint counterfactual scenario"""
        
        current_time = time.time()
        scenario_id = f"ethical_constraint_{decision_id}_{variant}"
        
        # Modify ethical constraints
        constraint_strength = 0.3 + (variant * 0.2)  # Varying ethical strictness
        changed_factors = {"ethical_constraint_strength": constraint_strength}
        
        confidence = max(0.5, 1.0 - abs(constraint_strength - 0.7) * 0.5)
        probability = 0.7 - (constraint_strength * 0.2)  # Stricter ethics = lower probability of action
        
        explanation_text = f"Ethical constraint scenario: With {'stricter' if constraint_strength > 0.7 else 'relaxed'} ethical constraints"
        
        impact_assessment = {
            "ethical_compliance": constraint_strength,
            "decision_freedom": 1.0 - constraint_strength,
            "stakeholder_acceptance": constraint_strength * 0.8
        }
        
        return CounterfactualScenario(
            scenario_id=scenario_id,
            original_decision_id=decision_id,
            changed_factors=changed_factors,
            predicted_outcome=f"Ethically-constrained decision: {constraint_strength:.1f} constraint strength",
            confidence=confidence,
            probability=probability,
            explanation=explanation_text,
            impact_assessment=impact_assessment,
            timestamp=current_time
        )
    
    def _generate_temporal_shift_scenario(self, decision_id: str, original_context: Dict[str, Any],
                                        original_decision: Any, variant: int) -> CounterfactualScenario:
        """Generate a temporal shift counterfactual scenario"""
        
        current_time = time.time()
        scenario_id = f"temporal_shift_{decision_id}_{variant}"
        
        # Modify timing
        time_shift = (variant - 2) * 0.5  # -1.0 to +1.0 (earlier to later)
        changed_factors = {"decision_timing": time_shift}
        
        confidence = max(0.4, 0.8 - abs(time_shift) * 0.3)
        probability = 0.6 + (abs(time_shift) * 0.2)  # Timing changes affect probability
        
        explanation_text = f"Temporal shift scenario: Decision made {'earlier' if time_shift < 0 else 'later'} by {abs(time_shift):.1f} time units"
        
        impact_assessment = {
            "timeliness": abs(time_shift) * 0.8,
            "opportunity_cost": abs(time_shift) * 0.6,
            "coordination_impact": abs(time_shift) * 0.7
        }
        
        return CounterfactualScenario(
            scenario_id=scenario_id,
            original_decision_id=decision_id,
            changed_factors=changed_factors,
            predicted_outcome=f"Time-shifted decision: {time_shift:.1f} timing adjustment",
            confidence=confidence,
            probability=probability,
            explanation=explanation_text,
            impact_assessment=impact_assessment,
            timestamp=current_time
        )
    
    def _generate_information_absence_scenario(self, decision_id: str, original_context: Dict[str, Any],
                                             original_decision: Any, variant: int) -> CounterfactualScenario:
        """Generate an information absence counterfactual scenario"""
        
        current_time = time.time()
        scenario_id = f"information_absence_{decision_id}_{variant}"
        
        # Remove information
        info_reduction = 0.2 + (variant * 0.2)  # 20% to 80% information removed
        changed_factors = {"information_completeness": 1.0 - info_reduction}
        
        confidence = max(0.2, 0.9 - info_reduction)
        probability = max(0.3, 0.8 - info_reduction * 0.5)
        
        explanation_text = f"Information absence scenario: With {info_reduction*100:.0f}% less information available"
        
        impact_assessment = {
            "decision_quality": info_reduction * 0.9,
            "uncertainty": info_reduction * 0.8,
            "risk_level": info_reduction * 0.7
        }
        
        return CounterfactualScenario(
            scenario_id=scenario_id,
            original_decision_id=decision_id,
            changed_factors=changed_factors,
            predicted_outcome=f"Information-limited decision: {(1-info_reduction)*100:.0f}% information",
            confidence=confidence,
            probability=probability,
            explanation=explanation_text,
            impact_assessment=impact_assessment,
            timestamp=current_time
        )
    
    def _calculate_confidence_distribution(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Calculate confidence distribution statistics"""
        confidences = [s.confidence for s in scenarios]
        
        return {
            "mean": np.mean(confidences),
            "std": np.std(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "quartiles": np.percentile(confidences, [25, 50, 75]).tolist()
        }
    
    def _analyze_outcome_variations(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Analyze variations in predicted outcomes"""
        outcomes = [str(s.predicted_outcome) for s in scenarios]
        unique_outcomes = len(set(outcomes))
        
        return {
            "total_outcomes": len(outcomes),
            "unique_outcomes": unique_outcomes,
            "outcome_diversity": unique_outcomes / max(len(outcomes), 1),
            "most_common_outcome": max(set(outcomes), key=outcomes.count) if outcomes else None
        }
    
    def _perform_sensitivity_analysis(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Perform sensitivity analysis on scenario factors"""
        
        factor_impacts = defaultdict(list)
        
        for scenario in scenarios:
            avg_impact = np.mean(list(scenario.impact_assessment.values()))
            for factor, value in scenario.changed_factors.items():
                factor_impacts[factor].append((value, avg_impact))
        
        sensitivity_results = {}
        for factor, impacts in factor_impacts.items():
            if len(impacts) > 1:
                values = [imp[0] for imp in impacts]
                impact_scores = [imp[1] for imp in impacts]
                
                # Calculate correlation
                correlation = np.corrcoef(values, impact_scores)[0, 1] if len(values) > 1 else 0
                
                sensitivity_results[factor] = {
                    "correlation": correlation,
                    "impact_range": (min(impact_scores), max(impact_scores)),
                    "sensitivity": abs(correlation)
                }
        
        return sensitivity_results
    
    def _analyze_changed_factors(self, scenarios: List[CounterfactualScenario]) -> Dict[str, Any]:
        """Analyze which factors are changed most frequently"""
        
        factor_counts = defaultdict(int)
        factor_impacts = defaultdict(list)
        
        for scenario in scenarios:
            for factor in scenario.changed_factors.keys():
                factor_counts[factor] += 1
                avg_impact = np.mean(list(scenario.impact_assessment.values()))
                factor_impacts[factor].append(avg_impact)
        
        factor_analysis = {}
        for factor, count in factor_counts.items():
            factor_analysis[factor] = {
                "frequency": count,
                "average_impact": np.mean(factor_impacts[factor]),
                "impact_variance": np.var(factor_impacts[factor])
            }
        
        return factor_analysis


class NaturalLanguageExplainer:
    """System for generating natural language explanations of decisions"""
    
    def __init__(self):
        self.explanation_templates = {
            "summary": "The system decided to {decision} with {confidence}% confidence based on {key_factors}.",
            "detailed": "After analyzing {num_factors} factors over {processing_time} seconds, the system chose {decision}. The key reasoning steps were: {reasoning_steps}. This decision has {confidence}% confidence.",
            "technical": "Decision ID {decision_id}: Processing {num_inputs} inputs through {num_reasoning_steps} reasoning steps resulted in decision '{decision}' with confidence score {confidence}. Primary factors: {technical_factors}.",
            "ethical": "From an ethical perspective, this decision scored {ethics_score}/1.0 on our compliance framework. {ethical_considerations} The decision aligns with {compliant_laws} ethical principles."
        }
        
        self.confidence_phrases = {
            (0.9, 1.0): ["very high confidence", "extremely confident", "highly certain"],
            (0.7, 0.9): ["high confidence", "confident", "reasonably certain"],
            (0.5, 0.7): ["moderate confidence", "somewhat confident", "cautiously optimistic"],
            (0.3, 0.5): ["low confidence", "uncertain", "tentative"],
            (0.0, 0.3): ["very low confidence", "highly uncertain", "speculative"]
        }
    
    def generate_explanation(self, decision_data: Dict[str, Any], 
                           explanation_type: str = "summary") -> NaturalLanguageExplanation:
        """Generate natural language explanation for a decision"""
        
        current_time = time.time()
        decision_id = decision_data.get("decision_id", "unknown")
        
        # Extract key information
        decision = str(decision_data.get("decision", "unknown decision"))
        confidence = decision_data.get("confidence", 0.5)
        reasoning_chain = decision_data.get("reasoning_chain", [])
        ethical_factors = decision_data.get("ethical_factors", [])
        
        # Generate explanation based on type
        explanation_text = self._generate_explanation_text(
            decision_data, explanation_type
        )
        
        # Extract key factors
        key_factors = self._extract_key_factors(decision_data)
        
        # Generate confidence phrases
        confidence_phrases = self._get_confidence_phrases(confidence)
        
        # Generate reasoning chain text
        reasoning_chain_text = self._generate_reasoning_chain_text(reasoning_chain)
        
        # Generate ethical considerations text
        ethical_considerations_text = self._generate_ethical_text(ethical_factors)
        
        # Generate alternative explanations
        alternative_explanations = self._generate_alternative_explanations(
            decision_data, explanation_type
        )
        
        # Calculate readability score
        readability_score = self._calculate_readability(explanation_text)
        
        return NaturalLanguageExplanation(
            decision_id=decision_id,
            explanation_type=explanation_type,
            explanation_text=explanation_text,
            key_factors=key_factors,
            confidence_phrases=confidence_phrases,
            reasoning_chain_text=reasoning_chain_text,
            ethical_considerations_text=ethical_considerations_text,
            alternative_explanations=alternative_explanations,
            readability_score=readability_score,
            timestamp=current_time
        )
    
    def _generate_explanation_text(self, decision_data: Dict[str, Any], 
                                 explanation_type: str) -> str:
        """Generate the main explanation text"""
        
        template = self.explanation_templates.get(explanation_type, self.explanation_templates["summary"])
        
        # Prepare template variables
        template_vars = {
            "decision": str(decision_data.get("decision", "proceed")),
            "confidence": int(decision_data.get("confidence", 0.5) * 100),
            "decision_id": decision_data.get("decision_id", "unknown"),
            "num_factors": len(decision_data.get("inputs", {})),
            "processing_time": decision_data.get("processing_duration", 0),
            "num_inputs": len(decision_data.get("inputs", {})),
            "num_reasoning_steps": len(decision_data.get("reasoning_chain", [])),
            "ethics_score": decision_data.get("overall_ethics_score", 0.5),
            "key_factors": ", ".join(self._extract_key_factors(decision_data)[:3]),
            "reasoning_steps": self._get_reasoning_summary(decision_data.get("reasoning_chain", [])),
            "technical_factors": self._get_technical_factors(decision_data),
            "ethical_considerations": self._get_ethical_summary(decision_data.get("ethical_factors", [])),
            "compliant_laws": self._get_compliant_laws(decision_data.get("ethical_factors", []))
        }
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return f"Decision made: {template_vars.get('decision', 'unknown')} with {template_vars.get('confidence', 50)}% confidence."
    
    def _extract_key_factors(self, decision_data: Dict[str, Any]) -> List[str]:
        """Extract key factors that influenced the decision"""
        factors = []
        
        # From inputs
        inputs = decision_data.get("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, (int, float)) and abs(value) > 0.1:
                factors.append(key)
            elif isinstance(value, bool) and value:
                factors.append(key)
            elif isinstance(value, str) and len(value) > 0:
                factors.append(key)
        
        # From reasoning chain
        reasoning_chain = decision_data.get("reasoning_chain", [])
        for step in reasoning_chain:
            if isinstance(step, dict) and step.get("confidence", 0) > 0.7:
                step_type = step.get("type", step.get("step_type", "unknown"))
                factors.append(f"{step_type}_reasoning")
        
        # From ethical factors
        ethical_factors = decision_data.get("ethical_factors", [])
        for factor in ethical_factors:
            if isinstance(factor, dict) and factor.get("compliance_score", 0) > 0.8:
                factors.append(factor.get("factor_name", "ethical_factor"))
        
        return factors[:10]  # Return top 10 factors
    
    def _get_confidence_phrases(self, confidence: float) -> List[str]:
        """Get appropriate confidence phrases"""
        for (min_conf, max_conf), phrases in self.confidence_phrases.items():
            if min_conf <= confidence < max_conf:
                return phrases
        
        return ["uncertain confidence level"]
    
    def _generate_reasoning_chain_text(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """Generate text describing the reasoning chain"""
        if not reasoning_chain:
            return "No detailed reasoning steps available."
        
        steps = []
        for i, step in enumerate(reasoning_chain[:5]):  # Limit to 5 steps
            if isinstance(step, dict):
                step_type = step.get("type", step.get("step_type", "step"))
                description = step.get("description", "processed information")
                confidence = step.get("confidence", 0.5)
                
                step_text = f"{i+1}. {step_type.title()}: {description} (confidence: {confidence:.1f})"
                steps.append(step_text)
        
        return " ".join(steps)
    
    def _generate_ethical_text(self, ethical_factors: List[Dict[str, Any]]) -> str:
        """Generate text describing ethical considerations"""
        if not ethical_factors:
            return "No specific ethical factors were evaluated."
        
        considerations = []
        for factor in ethical_factors[:3]:  # Limit to 3 factors
            if isinstance(factor, dict):
                factor_name = factor.get("factor_name", "ethical consideration")
                compliance_score = factor.get("compliance_score", 0.5)
                assessment = factor.get("assessment", "evaluated")
                
                consideration = f"{factor_name} ({assessment}, score: {compliance_score:.1f})"
                considerations.append(consideration)
        
        return "Key ethical considerations: " + ", ".join(considerations)
    
    def _generate_alternative_explanations(self, decision_data: Dict[str, Any], 
                                         current_type: str) -> List[str]:
        """Generate alternative explanations using different perspectives"""
        alternatives = []
        
        # Generate explanations with different types
        other_types = [t for t in self.explanation_templates.keys() if t != current_type]
        
        for alt_type in other_types[:2]:  # Limit to 2 alternatives
            try:
                alt_explanation = self._generate_explanation_text(decision_data, alt_type)
                alternatives.append(alt_explanation)
            except Exception as e:
                logger.warning(f"Failed to generate alternative explanation: {e}")
        
        return alternatives
    
    def _get_reasoning_summary(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """Get a brief summary of reasoning steps"""
        if not reasoning_chain:
            return "direct decision"
        
        step_types = []
        for step in reasoning_chain:
            if isinstance(step, dict):
                step_type = step.get("type", step.get("step_type", "processing"))
                if step_type not in step_types:
                    step_types.append(step_type)
        
        return " → ".join(step_types[:4])  # Limit to 4 steps
    
    def _get_technical_factors(self, decision_data: Dict[str, Any]) -> str:
        """Get technical factors summary"""
        factors = []
        
        inputs = decision_data.get("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, (int, float)):
                factors.append(f"{key}={value:.2f}")
        
        return ", ".join(factors[:5])  # Limit to 5 factors
    
    def _get_ethical_summary(self, ethical_factors: List[Dict[str, Any]]) -> str:
        """Get ethical factors summary"""
        if not ethical_factors:
            return "Standard ethical guidelines were applied."
        
        high_compliance = [f for f in ethical_factors if f.get("compliance_score", 0) > 0.8]
        if high_compliance:
            return f"High compliance with {len(high_compliance)} ethical principles."
        else:
            return "Ethical compliance was evaluated."
    
    def _get_compliant_laws(self, ethical_factors: List[Dict[str, Any]]) -> str:
        """Get compliant laws/principles"""
        if not ethical_factors:
            return "general"
        
        laws = []
        for factor in ethical_factors:
            if isinstance(factor, dict) and factor.get("compliance_score", 0) > 0.7:
                law_ref = factor.get("law_reference", "")
                if law_ref and law_ref not in laws:
                    laws.append(law_ref)
        
        return ", ".join(laws[:3]) if laws else "standard"
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        if not text:
            return 0.0
        
        # Count sentences (rough approximation)
        sentences = len([s for s in text.split('.') if s.strip()])
        
        # Count words
        words = len(text.split())
        
        # Count syllables (rough approximation)
        vowels = 'aeiouAEIOU'
        syllables = sum(1 for char in text if char in vowels)
        
        if sentences == 0 or words == 0:
            return 0.0
        
        # Simplified Flesch Reading Ease formula
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        # Flesch Reading Ease = 206.835 - 1.015 * ASL - 84.6 * ASW
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, score / 100.0))


class DecisionTransparencySystem:
    """Main system integrating all decision transparency features"""
    
    def __init__(self):
        self.attention_tracker = MultiHeadAttentionTracker()
        self.counterfactual_reasoner = CounterfactualReasoner()
        self.language_explainer = NaturalLanguageExplainer()
        
        self.transparency_logs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Decision Transparency System initialized")
    
    def analyze_decision_transparency(self, decision_id: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive transparency analysis for a decision"""
        
        current_time = time.time()
        
        # Generate attention analysis
        attention_map = self.attention_tracker.calculate_attention(
            decision_data.get("inputs", {}),
            {"decision_id": decision_id, "layer": "main"}
        )
        
        attention_viz = self.attention_tracker.visualize_attention(attention_map)
        
        # Generate counterfactual scenarios
        counterfactual_scenarios = self.counterfactual_reasoner.generate_counterfactual_scenarios(
            decision_id, decision_data.get("context", {}), decision_data.get("decision")
        )
        
        counterfactual_analysis = self.counterfactual_reasoner.analyze_counterfactual_impact(counterfactual_scenarios)
        counterfactual_viz = self.counterfactual_reasoner.get_scenario_visualization(counterfactual_scenarios)
        
        # Generate natural language explanations
        explanations = {}
        for explanation_type in ["summary", "detailed", "technical", "ethical"]:
            explanation = self.language_explainer.generate_explanation(decision_data, explanation_type)
            explanations[explanation_type] = explanation
        
        transparency_analysis = {
            "decision_id": decision_id,
            "timestamp": current_time,
            "attention_analysis": {
                "attention_map": asdict(attention_map),
                "visualization": attention_viz
            },
            "counterfactual_analysis": {
                "scenarios": [asdict(s) for s in counterfactual_scenarios],
                "impact_analysis": counterfactual_analysis,
                "visualization": counterfactual_viz
            },
            "natural_language_explanations": {
                explanation_type: asdict(explanation)
                for explanation_type, explanation in explanations.items()
            },
            "transparency_metrics": {
                "attention_entropy": attention_map.attention_entropy,
                "explanation_readability": np.mean([exp.readability_score for exp in explanations.values()]),
                "counterfactual_diversity": len(set(s.explanation for s in counterfactual_scenarios)),
                "overall_transparency_score": self._calculate_transparency_score(attention_map, counterfactual_scenarios, explanations)
            }
        }
        
        # Store analysis
        self.transparency_logs[decision_id] = transparency_analysis
        
        return transparency_analysis
    
    def get_transparency_dashboard(self, decision_ids: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive transparency dashboard"""
        
        current_time = time.time()
        
        if decision_ids is None:
            decision_ids = list(self.transparency_logs.keys())
        
        # Aggregate transparency metrics
        if decision_ids:
            logs = [self.transparency_logs[did] for did in decision_ids if did in self.transparency_logs]
            
            metrics = {
                "total_decisions_analyzed": len(logs),
                "average_attention_entropy": np.mean([log["transparency_metrics"]["attention_entropy"] for log in logs]),
                "average_readability": np.mean([log["transparency_metrics"]["explanation_readability"] for log in logs]),
                "average_transparency_score": np.mean([log["transparency_metrics"]["overall_transparency_score"] for log in logs]),
                "decisions_with_high_transparency": len([log for log in logs if log["transparency_metrics"]["overall_transparency_score"] > 0.7])
            }
        else:
            metrics = {"total_decisions_analyzed": 0}
        
        dashboard = {
            "timestamp": current_time,
            "summary_metrics": metrics,
            "recent_analyses": [
                {
                    "decision_id": log["decision_id"],
                    "timestamp": log["timestamp"],
                    "transparency_score": log["transparency_metrics"]["overall_transparency_score"],
                    "attention_entropy": log["transparency_metrics"]["attention_entropy"]
                }
                for log in sorted(self.transparency_logs.values(), key=lambda x: x["timestamp"], reverse=True)[:10]
            ],
            "transparency_trends": self._calculate_transparency_trends(),
            "attention_patterns": self._analyze_attention_patterns(),
            "explanation_quality": self._analyze_explanation_quality()
        }
        
        return dashboard
    
    def _calculate_transparency_score(self, attention_map: AttentionMap, 
                                    scenarios: List[CounterfactualScenario],
                                    explanations: Dict[str, NaturalLanguageExplanation]) -> float:
        """Calculate overall transparency score"""
        
        # Attention clarity score (lower entropy = higher clarity)
        attention_score = 1.0 - attention_map.attention_entropy
        
        # Counterfactual diversity score
        if scenarios:
            unique_explanations = len(set(s.explanation for s in scenarios))
            counterfactual_score = min(1.0, unique_explanations / 5.0)  # Normalize to max 5
        else:
            counterfactual_score = 0.0
        
        # Explanation quality score
        readability_scores = [exp.readability_score for exp in explanations.values()]
        explanation_score = np.mean(readability_scores) if readability_scores else 0.0
        
        # Weighted combination
        overall_score = (attention_score * 0.3 + counterfactual_score * 0.3 + explanation_score * 0.4)
        
        return max(0.0, min(1.0, overall_score))
    
    def _calculate_transparency_trends(self) -> Dict[str, Any]:
        """Calculate transparency trends over time"""
        if not self.transparency_logs:
            return {"message": "No transparency data available"}
        
        # Sort logs by timestamp
        sorted_logs = sorted(self.transparency_logs.values(), key=lambda x: x["timestamp"])
        
        # Calculate trends
        transparency_scores = [log["transparency_metrics"]["overall_transparency_score"] for log in sorted_logs]
        attention_entropies = [log["transparency_metrics"]["attention_entropy"] for log in sorted_logs]
        
        if len(transparency_scores) > 1:
            transparency_trend = "improving" if transparency_scores[-1] > transparency_scores[0] else "declining"
            attention_trend = "improving" if attention_entropies[-1] < attention_entropies[0] else "declining"  # Lower entropy is better
        else:
            transparency_trend = "stable"
            attention_trend = "stable"
        
        return {
            "transparency_trend": transparency_trend,
            "attention_trend": attention_trend,
            "latest_score": transparency_scores[-1] if transparency_scores else 0,
            "score_change": transparency_scores[-1] - transparency_scores[0] if len(transparency_scores) > 1 else 0
        }
    
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze common attention patterns"""
        if not self.transparency_logs:
            return {"message": "No attention data available"}
        
        # Collect all attention weights
        all_factors = defaultdict(list)
        
        for log in self.transparency_logs.values():
            attention_map = log["attention_analysis"]["attention_map"]
            for weight in attention_map["attention_weights"]:
                all_factors[weight["factor_name"]].append(weight["weight"])
        
        # Analyze patterns
        pattern_analysis = {}
        for factor, weights in all_factors.items():
            pattern_analysis[factor] = {
                "frequency": len(weights),
                "average_weight": np.mean(weights),
                "weight_variance": np.var(weights),
                "max_weight": np.max(weights)
            }
        
        # Sort by frequency
        sorted_patterns = sorted(pattern_analysis.items(), key=lambda x: x[1]["frequency"], reverse=True)
        
        return {
            "most_attended_factors": sorted_patterns[:10],
            "total_unique_factors": len(all_factors),
            "average_factors_per_decision": np.mean([len(attention_map["attention_weights"]) for log in self.transparency_logs.values() for attention_map in [log["attention_analysis"]["attention_map"]]])
        }
    
    def _analyze_explanation_quality(self) -> Dict[str, Any]:
        """Analyze quality of natural language explanations"""
        if not self.transparency_logs:
            return {"message": "No explanation data available"}
        
        readability_scores = []
        explanation_lengths = []
        
        for log in self.transparency_logs.values():
            explanations = log["natural_language_explanations"]
            for exp_type, explanation in explanations.items():
                readability_scores.append(explanation["readability_score"])
                explanation_lengths.append(len(explanation["explanation_text"]))
        
        if readability_scores:
            quality_analysis = {
                "average_readability": np.mean(readability_scores),
                "readability_variance": np.var(readability_scores),
                "average_explanation_length": np.mean(explanation_lengths),
                "high_readability_explanations": len([s for s in readability_scores if s > 0.7]),
                "total_explanations": len(readability_scores)
            }
        else:
            quality_analysis = {"message": "No explanation quality data available"}
        
        return quality_analysis