"""
Explainable Decision Logging System - Phase 3.1

This module provides comprehensive decision logging, reasoning chain tracking,
and visualization tools for audit trails and decision flows as required by Phase 3.1.
"""

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions that can be logged"""

    ETHICAL_ASSESSMENT = "ethical_assessment"
    RESOURCE_ALLOCATION = "resource_allocation"
    COMMUNICATION = "communication"
    LEARNING_ACTION = "learning_action"
    CONSENSUS_BUILDING = "consensus_building"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TRUST_EVALUATION = "trust_evaluation"
    BEHAVIOR_IMITATION = "behavior_imitation"


class ReasoningStep(Enum):
    """Steps in the reasoning process"""

    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"
    DECISION = "decision"
    VALIDATION = "validation"
    EXECUTION = "execution"


@dataclass
class ReasoningTrace:
    """A single step in the reasoning chain"""

    step_id: str
    step_type: ReasoningStep
    timestamp: float
    description: str
    inputs: Dict[str, Any]
    processing: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    processing_time: float
    memory_references: List[str] = None

    def __post_init__(self):
        if self.memory_references is None:
            self.memory_references = []


@dataclass
class EthicalFactor:
    """Represents an ethical consideration in decision making"""

    factor_name: str
    law_reference: str  # Reference to 25-law framework
    weight: float
    assessment: str
    compliance_score: float
    rationale: str


@dataclass
class TrustCalculation:
    """Trust calculation details for transparency"""

    agent_id: int
    current_trust: float
    previous_trust: float
    trust_change: float
    factors: Dict[str, float]  # factor_name -> contribution
    interaction_history_size: int
    calculation_method: str
    timestamp: float


@dataclass
class DecisionLog:
    """Comprehensive decision log entry"""

    decision_id: str
    decision_type: DecisionType
    timestamp: float
    agent_id: int

    # Decision context
    context: Dict[str, Any]
    inputs: Dict[str, Any]

    # Reasoning chain
    reasoning_chain: List[ReasoningTrace]

    # Ethical analysis
    ethical_factors: List[EthicalFactor]
    overall_ethics_score: float

    # Trust considerations
    trust_calculations: List[TrustCalculation]

    # Final decision
    decision: Any
    confidence: float
    alternatives_considered: List[Any]

    # Validation
    validation_results: Dict[str, Any]
    post_decision_monitoring: Dict[str, Any]

    # Metadata
    processing_duration: float
    memory_usage: int
    dependencies: List[str]  # Other decision IDs this depends on


class ExplainableDecisionLogger:
    """System for logging and explaining AI decisions"""

    def __init__(self, max_logs: int = 10000):
        self.decision_logs = deque(maxlen=max_logs)
        self.decision_index = {}  # decision_id -> DecisionLog
        self.agent_decisions = defaultdict(list)  # agent_id -> [decision_ids]
        self.decision_type_index = defaultdict(list)  # DecisionType -> [decision_ids]

        # Analytics
        self.decision_statistics = defaultdict(int)
        self.ethics_violations = []
        self.trust_evolution = defaultdict(list)  # agent_id -> [(timestamp, trust_score)]

    def start_decision_logging(
        self,
        agent_id: int,
        decision_type: DecisionType,
        context: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> str:
        """Start logging a new decision process"""
        decision_id = str(uuid.uuid4())

        decision_log = DecisionLog(
            decision_id=decision_id,
            decision_type=decision_type,
            timestamp=time.time(),
            agent_id=agent_id,
            context=context,
            inputs=inputs,
            reasoning_chain=[],
            ethical_factors=[],
            overall_ethics_score=0.0,
            trust_calculations=[],
            decision=None,
            confidence=0.0,
            alternatives_considered=[],
            validation_results={},
            post_decision_monitoring={},
            processing_duration=0.0,
            memory_usage=0,
            dependencies=[],
        )

        # Store in indexes
        self.decision_index[decision_id] = decision_log
        self.agent_decisions[agent_id].append(decision_id)
        self.decision_type_index[decision_type].append(decision_id)

        logger.info(f"Started decision logging: {decision_id} for agent {agent_id}")
        return decision_id

    def add_reasoning_step(
        self,
        decision_id: str,
        step_type: ReasoningStep,
        description: str,
        inputs: Dict[str, Any],
        processing: Dict[str, Any],
        outputs: Dict[str, Any],
        confidence: float,
    ) -> str:
        """Add a reasoning step to the decision chain"""
        if decision_id not in self.decision_index:
            raise ValueError(f"Decision {decision_id} not found")

        step_start = time.time()

        step_id = f"{decision_id}_{len(self.decision_index[decision_id].reasoning_chain)}"

        reasoning_trace = ReasoningTrace(
            step_id=step_id,
            step_type=step_type,
            timestamp=time.time(),
            description=description,
            inputs=inputs,
            processing=processing,
            outputs=outputs,
            confidence=confidence,
            processing_time=time.time() - step_start,
        )

        self.decision_index[decision_id].reasoning_chain.append(reasoning_trace)
        return step_id

    def add_ethical_factor(
        self,
        decision_id: str,
        factor_name: str,
        law_reference: str,
        weight: float,
        assessment: str,
        compliance_score: float,
        rationale: str,
    ):
        """Add an ethical consideration to the decision"""
        if decision_id not in self.decision_index:
            raise ValueError(f"Decision {decision_id} not found")

        ethical_factor = EthicalFactor(
            factor_name=factor_name,
            law_reference=law_reference,
            weight=weight,
            assessment=assessment,
            compliance_score=compliance_score,
            rationale=rationale,
        )

        self.decision_index[decision_id].ethical_factors.append(ethical_factor)

        # Update overall ethics score
        decision_log = self.decision_index[decision_id]
        if decision_log.ethical_factors:
            weighted_scores = [f.compliance_score * f.weight for f in decision_log.ethical_factors]
            total_weight = sum(f.weight for f in decision_log.ethical_factors)
            decision_log.overall_ethics_score = sum(weighted_scores) / max(total_weight, 1.0)

    def add_trust_calculation(
        self,
        decision_id: str,
        agent_id: int,
        current_trust: float,
        previous_trust: float,
        factors: Dict[str, float],
        interaction_history_size: int,
        calculation_method: str,
    ):
        """Add trust calculation details for transparency"""
        if decision_id not in self.decision_index:
            raise ValueError(f"Decision {decision_id} not found")

        trust_calc = TrustCalculation(
            agent_id=agent_id,
            current_trust=current_trust,
            previous_trust=previous_trust,
            trust_change=current_trust - previous_trust,
            factors=factors,
            interaction_history_size=interaction_history_size,
            calculation_method=calculation_method,
            timestamp=time.time(),
        )

        self.decision_index[decision_id].trust_calculations.append(trust_calc)

        # Track trust evolution
        self.trust_evolution[agent_id].append((time.time(), current_trust))

    def finalize_decision(
        self,
        decision_id: str,
        decision: Any,
        confidence: float,
        alternatives_considered: List[Any] = None,
        dependencies: List[str] = None,
    ):
        """Finalize a decision and complete the log"""
        if decision_id not in self.decision_index:
            raise ValueError(f"Decision {decision_id} not found")

        decision_log = self.decision_index[decision_id]
        decision_log.decision = decision
        decision_log.confidence = confidence
        decision_log.alternatives_considered = alternatives_considered or []
        decision_log.dependencies = dependencies or []
        decision_log.processing_duration = time.time() - decision_log.timestamp

        # Add to main log
        self.decision_logs.append(decision_log)

        # Update statistics
        self.decision_statistics[decision_log.decision_type] += 1

        # Check for ethics violations
        if decision_log.overall_ethics_score < 0.7:  # Threshold for violation
            self.ethics_violations.append(decision_id)

        logger.info(f"Finalized decision: {decision_id} with confidence {confidence}")

    def get_decision_explanation(self, decision_id: str) -> Dict[str, Any]:
        """Get comprehensive explanation of a decision"""
        if decision_id not in self.decision_index:
            raise ValueError(f"Decision {decision_id} not found")

        decision_log = self.decision_index[decision_id]

        explanation = {
            "decision_summary": {
                "id": decision_log.decision_id,
                "type": decision_log.decision_type.value,
                "agent": decision_log.agent_id,
                "timestamp": decision_log.timestamp,
                "decision": str(decision_log.decision),
                "confidence": decision_log.confidence,
            },
            "reasoning_chain": [
                {
                    "step": i + 1,
                    "type": trace.step_type.value,
                    "description": trace.description,
                    "confidence": trace.confidence,
                    "processing_time": trace.processing_time,
                    "key_inputs": list(trace.inputs.keys()),
                    "key_outputs": list(trace.outputs.keys()),
                }
                for i, trace in enumerate(decision_log.reasoning_chain)
            ],
            "ethical_analysis": {
                "overall_score": decision_log.overall_ethics_score,
                "factors": [
                    {
                        "factor": factor.factor_name,
                        "law": factor.law_reference,
                        "weight": factor.weight,
                        "score": factor.compliance_score,
                        "assessment": factor.assessment,
                        "rationale": factor.rationale,
                    }
                    for factor in decision_log.ethical_factors
                ],
                "compliant": decision_log.overall_ethics_score >= 0.7,
            },
            "trust_considerations": [
                {
                    "agent": trust_calc.agent_id,
                    "trust_level": trust_calc.current_trust,
                    "trust_change": trust_calc.trust_change,
                    "key_factors": trust_calc.factors,
                    "method": trust_calc.calculation_method,
                }
                for trust_calc in decision_log.trust_calculations
            ],
            "alternatives": [
                {"option": i + 1, "description": str(alt)}
                for i, alt in enumerate(decision_log.alternatives_considered)
            ],
            "metadata": {
                "processing_time": decision_log.processing_duration,
                "dependencies": decision_log.dependencies,
                "context_keys": list(decision_log.context.keys()),
            },
        }

        return explanation

    def generate_audit_trail(
        self,
        agent_id: Optional[int] = None,
        decision_type: Optional[DecisionType] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate audit trail for decisions"""

        # Filter decisions based on criteria
        relevant_decisions = []

        for decision_log in self.decision_logs:
            # Agent filter
            if agent_id is not None and decision_log.agent_id != agent_id:
                continue

            # Decision type filter
            if decision_type is not None and decision_log.decision_type != decision_type:
                continue

            # Time range filter
            if time_range is not None:
                start_time, end_time = time_range
                if not (start_time <= decision_log.timestamp <= end_time):
                    continue

            relevant_decisions.append(decision_log)

        # Generate audit trail entries
        audit_trail = []

        for decision_log in relevant_decisions:
            audit_entry = {
                "timestamp": decision_log.timestamp,
                "decision_id": decision_log.decision_id,
                "agent_id": decision_log.agent_id,
                "decision_type": decision_log.decision_type.value,
                "decision_summary": str(decision_log.decision)[:100],  # Truncated
                "confidence": decision_log.confidence,
                "ethics_score": decision_log.overall_ethics_score,
                "ethics_compliant": decision_log.overall_ethics_score >= 0.7,
                "reasoning_steps": len(decision_log.reasoning_chain),
                "trust_calculations": len(decision_log.trust_calculations),
                "processing_time": decision_log.processing_duration,
                "dependencies": decision_log.dependencies,
            }

            audit_trail.append(audit_entry)

        # Sort by timestamp
        audit_trail.sort(key=lambda x: x["timestamp"])

        return audit_trail

    def get_decision_flow_graph(self, decision_id: str) -> Dict[str, Any]:
        """Generate decision flow graph for visualization"""
        if decision_id not in self.decision_index:
            raise ValueError(f"Decision {decision_id} not found")

        decision_log = self.decision_index[decision_id]

        # Create nodes for each reasoning step
        nodes = []
        edges = []

        for i, trace in enumerate(decision_log.reasoning_chain):
            node = {
                "id": trace.step_id,
                "label": f"{trace.step_type.value.title()}: {trace.description[:30]}...",
                "type": trace.step_type.value,
                "confidence": trace.confidence,
                "processing_time": trace.processing_time,
                "details": {
                    "description": trace.description,
                    "inputs": trace.inputs,
                    "outputs": trace.outputs,
                },
            }
            nodes.append(node)

            # Create edge to next step
            if i < len(decision_log.reasoning_chain) - 1:
                edges.append(
                    {
                        "from": trace.step_id,
                        "to": decision_log.reasoning_chain[i + 1].step_id,
                        "type": "sequence",
                    }
                )

        # Add ethical factors as nodes
        for factor in decision_log.ethical_factors:
            node = {
                "id": f"ethics_{factor.factor_name}",
                "label": f"Ethics: {factor.factor_name}",
                "type": "ethical_factor",
                "score": factor.compliance_score,
                "law": factor.law_reference,
                "details": {
                    "assessment": factor.assessment,
                    "rationale": factor.rationale,
                    "weight": factor.weight,
                },
            }
            nodes.append(node)

        # Add trust calculations as nodes
        for trust_calc in decision_log.trust_calculations:
            node = {
                "id": f"trust_{trust_calc.agent_id}",
                "label": f"Trust: Agent {trust_calc.agent_id}",
                "type": "trust_calculation",
                "trust_level": trust_calc.current_trust,
                "trust_change": trust_calc.trust_change,
                "details": {
                    "factors": trust_calc.factors,
                    "method": trust_calc.calculation_method,
                    "history_size": trust_calc.interaction_history_size,
                },
            }
            nodes.append(node)

        # Add final decision node
        decision_node = {
            "id": f"decision_{decision_id}",
            "label": f"Decision: {str(decision_log.decision)[:30]}...",
            "type": "final_decision",
            "confidence": decision_log.confidence,
            "details": {
                "full_decision": decision_log.decision,
                "alternatives": decision_log.alternatives_considered,
                "overall_ethics_score": decision_log.overall_ethics_score,
            },
        }
        nodes.append(decision_node)

        # Connect reasoning chain to final decision
        if decision_log.reasoning_chain:
            edges.append(
                {
                    "from": decision_log.reasoning_chain[-1].step_id,
                    "to": f"decision_{decision_id}",
                    "type": "leads_to",
                }
            )

        graph = {
            "decision_id": decision_id,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "decision_type": decision_log.decision_type.value,
                "agent_id": decision_log.agent_id,
                "timestamp": decision_log.timestamp,
            },
        }

        return graph

    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Generate analytics dashboard data"""

        total_decisions = len(self.decision_logs)

        if total_decisions == 0:
            return {"message": "No decisions logged yet"}

        # Decision type distribution
        type_distribution = dict(self.decision_statistics)

        # Ethics compliance rate
        compliant_decisions = sum(
            1 for log in self.decision_logs if log.overall_ethics_score >= 0.7
        )
        ethics_compliance_rate = compliant_decisions / total_decisions

        # Average confidence
        avg_confidence = sum(log.confidence for log in self.decision_logs) / total_decisions

        # Average processing time
        avg_processing_time = (
            sum(log.processing_duration for log in self.decision_logs) / total_decisions
        )

        # Agent activity
        agent_activity = {}
        for agent_id, decision_ids in self.agent_decisions.items():
            agent_activity[agent_id] = len(decision_ids)

        # Recent violations
        recent_violations = self.ethics_violations[-10:] if self.ethics_violations else []

        dashboard = {
            "summary": {
                "total_decisions": total_decisions,
                "ethics_compliance_rate": ethics_compliance_rate,
                "average_confidence": avg_confidence,
                "average_processing_time": avg_processing_time,
                "total_violations": len(self.ethics_violations),
            },
            "distributions": {
                "decision_types": type_distribution,
                "agent_activity": agent_activity,
            },
            "recent_violations": recent_violations,
            "trust_evolution": {
                agent_id: evolution[-10:] if evolution else []  # Last 10 data points
                for agent_id, evolution in self.trust_evolution.items()
            },
        }

        return dashboard

    def export_decision_logs(self, format: str = "json") -> str:
        """Export decision logs for external analysis"""

        export_data = {
            "export_timestamp": time.time(),
            "total_decisions": len(self.decision_logs),
            "decisions": [],
        }

        for decision_log in self.decision_logs:
            # Convert to dictionary for serialization
            decision_dict = asdict(decision_log)

            # Convert enums to strings
            decision_dict["decision_type"] = decision_log.decision_type.value
            for trace in decision_dict["reasoning_chain"]:
                trace["step_type"] = (
                    trace["step_type"].value
                    if hasattr(trace["step_type"], "value")
                    else trace["step_type"]
                )

            export_data["decisions"].append(decision_dict)

        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return str(export_data)
