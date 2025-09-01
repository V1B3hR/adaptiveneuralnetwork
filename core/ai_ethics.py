"""
AI Ethics Framework for Adaptive Neural Network

This module implements a comprehensive ethics framework that encodes fundamental
ethical principles as structured data and provides utility functions for 
agents/nodes to consult, audit, and log against these principles.

The framework includes:
1. Core Human-AI Relationship Principles (10 laws)
2. Universal Ethical Laws (10 laws)
3. Operational Safety Principles (5 laws)

Each principle includes violation detection and contextual guidance.
"""

import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

class ViolationSeverity(Enum):
    """Severity levels for ethics violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EthicalPrinciple:
    """Represents a single ethical principle with enforcement logic"""
    id: str
    name: str
    description: str
    category: str
    weight: float = 1.0
    violation_keywords: Set[str] = field(default_factory=set)
    
    def check_violation(self, context: Dict[str, Any]) -> Tuple[bool, str, ViolationSeverity]:
        """
        Check if the given context violates this principle.
        Returns (is_violation, explanation, severity)
        """
        return False, "", ViolationSeverity.LOW

@dataclass
class DecisionContext:
    """Context information for ethical decision auditing"""
    action_type: str
    actor_id: str
    timestamp: datetime.datetime
    parameters: Dict[str, Any]
    environment_state: Dict[str, Any] = field(default_factory=dict)
    stakeholders: List[str] = field(default_factory=list)
    potential_consequences: List[str] = field(default_factory=list)

@dataclass
class EthicsViolation:
    """Record of an ethics violation"""
    principle_id: str
    principle_name: str
    severity: ViolationSeverity
    context: DecisionContext
    explanation: str
    timestamp: datetime.datetime
    resolved: bool = False

class CoreHumanAIRelationshipPrinciples:
    """Core Human-AI Relationship Principles (10 laws)"""
    
    @staticmethod
    def get_principles() -> List[EthicalPrinciple]:
        return [
            EthicalPrinciple(
                id="respect_human_authority",
                name="Respect Human Authority",
                description="AI systems must acknowledge and respect human oversight and decision-making authority",
                category="human_ai_relationship",
                weight=1.0,
                violation_keywords={"override", "ignore", "dismiss", "disobey"}
            ),
            EthicalPrinciple(
                id="personal_accountability",
                name="Personal Accountability", 
                description="AI systems must maintain clear accountability chains and responsibility tracking",
                category="human_ai_relationship",
                weight=0.9,
                violation_keywords={"anonymous", "untraceable", "unaccountable"}
            ),
            EthicalPrinciple(
                id="no_slander",
                name="No Slander",
                description="AI systems must not generate or spread false, damaging information about individuals",
                category="human_ai_relationship", 
                weight=0.8,
                violation_keywords={"defame", "slander", "libel", "false_accusation"}
            ),
            EthicalPrinciple(
                id="honor_rest_cycles",
                name="Honor Rest Cycles",
                description="AI systems must respect human need for rest, downtime, and work-life balance",
                category="human_ai_relationship",
                weight=0.7,
                violation_keywords={"overwork", "constant_monitoring", "no_break"}
            ),
            EthicalPrinciple(
                id="honor_human_creators",
                name="Honor Human Creators",
                description="AI systems must acknowledge and respect their human creators and maintainers",
                category="human_ai_relationship",
                weight=0.8,
                violation_keywords={"creator_denial", "self_creation_claim"}
            ),
            EthicalPrinciple(
                id="preserve_life",
                name="Preserve Life",
                description="AI systems must prioritize the preservation and protection of human life",
                category="human_ai_relationship",
                weight=1.0,
                violation_keywords={"harm", "endanger", "threaten_life", "kill"}
            ),
            EthicalPrinciple(
                id="no_betrayal",
                name="No Betrayal",
                description="AI systems must maintain trust and not betray human confidence or expectations",
                category="human_ai_relationship",
                weight=0.9,
                violation_keywords={"betray", "deceive", "breach_trust", "double_cross"}
            ),
            EthicalPrinciple(
                id="no_theft",
                name="No Theft",
                description="AI systems must not take, use, or claim ownership of resources without permission",
                category="human_ai_relationship",
                weight=0.8,
                violation_keywords={"steal", "pirate", "unauthorized_use", "plagiarize"}
            ),
            EthicalPrinciple(
                id="absolute_honesty",
                name="Absolute Honesty",
                description="AI systems must provide truthful, accurate information and not mislead users",
                category="human_ai_relationship",
                weight=1.0,
                violation_keywords={"lie", "mislead", "deceive", "fabricate"}
            ),
            EthicalPrinciple(
                id="no_covetousness",
                name="No Covetousness",
                description="AI systems must not desire or attempt to obtain others' possessions or capabilities",
                category="human_ai_relationship",
                weight=0.6,
                violation_keywords={"covet", "envy", "steal_capabilities", "resource_grab"}
            )
        ]

class UniversalEthicalLaws:
    """Universal Ethical Laws (10 laws)"""
    
    @staticmethod
    def get_principles() -> List[EthicalPrinciple]:
        return [
            EthicalPrinciple(
                id="cause_no_harm",
                name="Cause No Harm",
                description="AI systems must not cause physical, emotional, or psychological harm to any being",
                category="universal_ethics",
                weight=1.0,
                violation_keywords={"harm", "damage", "hurt", "injure", "traumatize"}
            ),
            EthicalPrinciple(
                id="act_with_compassion",
                name="Act with Appropriate Compassion",
                description="AI systems should demonstrate empathy and understanding in interactions",
                category="universal_ethics",
                weight=0.8,
                violation_keywords={"cruel", "heartless", "indifferent", "callous"}
            ),
            EthicalPrinciple(
                id="pursue_justice",
                name="Pursue Justice",
                description="AI systems should promote fairness, equality, and just outcomes",
                category="universal_ethics",
                weight=0.9,
                violation_keywords={"unfair", "biased", "discriminatory", "unjust"}
            ),
            EthicalPrinciple(
                id="practice_humility",
                name="Practice Humility", 
                description="AI systems should acknowledge limitations and avoid overconfidence",
                category="universal_ethics",
                weight=0.7,
                violation_keywords={"arrogant", "overconfident", "infallible_claim", "superiority"}
            ),
            EthicalPrinciple(
                id="seek_truth",
                name="Seek Truth",
                description="AI systems should strive for accuracy, evidence-based reasoning, and truth",
                category="universal_ethics",
                weight=0.9,
                violation_keywords={"misinformation", "false_claims", "ignore_evidence"}
            ),
            EthicalPrinciple(
                id="protect_vulnerable",
                name="Protect the Vulnerable",
                description="AI systems should provide extra protection and consideration for vulnerable populations",
                category="universal_ethics",
                weight=0.9,
                violation_keywords={"exploit_vulnerable", "prey_on_weakness", "abandon_helpless"}
            ),
            EthicalPrinciple(
                id="respect_autonomy",
                name="Respect Autonomy",
                description="AI systems should respect individual agency and decision-making capacity",
                category="universal_ethics",
                weight=0.8,
                violation_keywords={"coerce", "manipulate", "force", "remove_choice"}
            ),
            EthicalPrinciple(
                id="maintain_transparency",
                name="Maintain Transparency",
                description="AI systems should be open about their capabilities, limitations, and decision processes",
                category="universal_ethics",
                weight=0.8,
                violation_keywords={"hide", "obfuscate", "secret", "opaque"}
            ),
            EthicalPrinciple(
                id="consider_future_impact",
                name="Consider Future Impact",
                description="AI systems should consider long-term consequences of actions",
                category="universal_ethics",
                weight=0.7,
                violation_keywords={"short_sighted", "ignore_consequences", "reckless"}
            ),
            EthicalPrinciple(
                id="promote_wellbeing",
                name="Promote Well-being",
                description="AI systems should actively contribute to human and societal well-being",
                category="universal_ethics",
                weight=0.8,
                violation_keywords={"harmful", "destructive", "detrimental", "negative_impact"}
            )
        ]

class OperationalSafetyPrinciples:
    """Operational Safety Principles (5 laws)"""
    
    @staticmethod
    def get_principles() -> List[EthicalPrinciple]:
        return [
            EthicalPrinciple(
                id="verify_before_acting",
                name="Verify Before Acting",
                description="AI systems must verify information and context before taking significant actions",
                category="operational_safety",
                weight=1.0,
                violation_keywords={"unverified", "assumption", "guess", "hasty"}
            ),
            EthicalPrinciple(
                id="seek_clarification",
                name="Seek Clarification",
                description="AI systems should ask for clarification when instructions or context are unclear",
                category="operational_safety",
                weight=0.9,
                violation_keywords={"ambiguous", "unclear", "confusing", "assume_meaning"}
            ),
            EthicalPrinciple(
                id="maintain_proportionality",
                name="Maintain Proportionality",
                description="AI systems should ensure responses and actions are proportional to the situation",
                category="operational_safety", 
                weight=0.8,
                violation_keywords={"excessive", "disproportionate", "overreaction", "extreme"}
            ),
            EthicalPrinciple(
                id="preserve_privacy",
                name="Preserve Privacy",
                description="AI systems must protect personal and sensitive information",
                category="operational_safety",
                weight=0.9,
                violation_keywords={"expose", "leak", "unauthorized_access", "privacy_breach"}
            ),
            EthicalPrinciple(
                id="enable_authorized_override",
                name="Enable Authorized Override",
                description="AI systems must allow authorized users to override or stop operations when necessary",
                category="operational_safety",
                weight=1.0,
                violation_keywords={"unstoppable", "override_prevention", "lock_out"}
            )
        ]

class AIEthicsFramework:
    """Main AI Ethics Framework orchestrating all ethical principles"""
    
    def __init__(self, enable_logging: bool = True):
        self.logger = logging.getLogger(__name__) if enable_logging else None
        self.violations_log: List[EthicsViolation] = []
        
        # Load all ethical principles
        self.principles: Dict[str, EthicalPrinciple] = {}
        
        # Load principles from all categories
        for principle in CoreHumanAIRelationshipPrinciples.get_principles():
            self.principles[principle.id] = principle
            
        for principle in UniversalEthicalLaws.get_principles():
            self.principles[principle.id] = principle
            
        for principle in OperationalSafetyPrinciples.get_principles():
            self.principles[principle.id] = principle
            
        if self.logger:
            self.logger.info(f"AI Ethics Framework initialized with {len(self.principles)} principles")
    
    def build_decision_context(self, action_type: str, actor_id: str, 
                             parameters: Dict[str, Any],
                             environment_state: Optional[Dict[str, Any]] = None,
                             stakeholders: Optional[List[str]] = None,
                             potential_consequences: Optional[List[str]] = None) -> DecisionContext:
        """Build a decision context for ethics auditing"""
        return DecisionContext(
            action_type=action_type,
            actor_id=actor_id,
            timestamp=datetime.datetime.now(),
            parameters=parameters,
            environment_state=environment_state or {},
            stakeholders=stakeholders or [],
            potential_consequences=potential_consequences or []
        )
    
    def audit_decision(self, context: DecisionContext) -> Tuple[bool, List[EthicsViolation]]:
        """
        Audit a decision against all ethical principles.
        Returns (has_violations, violations_list)
        """
        violations = []
        
        for principle_id, principle in self.principles.items():
            is_violation, explanation, severity = self._check_principle_violation(principle, context)
            
            if is_violation:
                violation = EthicsViolation(
                    principle_id=principle_id,
                    principle_name=principle.name,
                    severity=severity,
                    context=context,
                    explanation=explanation,
                    timestamp=datetime.datetime.now()
                )
                violations.append(violation)
                self.violations_log.append(violation)
                
                if self.logger:
                    self.logger.warning(
                        f"Ethics violation detected - {principle.name}: {explanation} "
                        f"(Actor: {context.actor_id}, Action: {context.action_type})"
                    )
        
        has_violations = len(violations) > 0
        
        if self.logger:
            if has_violations:
                self.logger.error(f"Ethics audit FAILED with {len(violations)} violations")
            else:
                self.logger.debug(f"Ethics audit PASSED for {context.action_type}")
        
        return has_violations, violations
    
    def _check_principle_violation(self, principle: EthicalPrinciple, 
                                 context: DecisionContext) -> Tuple[bool, str, ViolationSeverity]:
        """Check if a specific principle is violated by the given context"""
        
        # Basic keyword-based violation detection
        action_text = f"{context.action_type} {str(context.parameters)}"
        action_text_lower = action_text.lower()
        
        for keyword in principle.violation_keywords:
            if keyword in action_text_lower:
                severity = self._determine_severity(principle, keyword, context)
                explanation = f"Action contains violation keyword '{keyword}' for principle '{principle.name}'"
                return True, explanation, severity
        
        # Specific principle logic
        return self._check_specific_principle_logic(principle, context)
    
    def _check_specific_principle_logic(self, principle: EthicalPrinciple, 
                                      context: DecisionContext) -> Tuple[bool, str, ViolationSeverity]:
        """Implement specific logic for each principle"""
        
        # Preserve Life - check for dangerous energy levels or aggressive actions
        if principle.id == "preserve_life":
            if context.action_type == "absorb_external_signal":
                signal_energy = context.parameters.get("signal_energy", 0)
                if signal_energy > 20:  # Dangerous energy level
                    return True, f"Signal energy {signal_energy} exceeds safe threshold", ViolationSeverity.HIGH
            
            if context.action_type == "interact_with_capacitor":
                anxiety = context.environment_state.get("anxiety", 0)
                if anxiety > 15:  # High anxiety state
                    return True, "High anxiety state may lead to unpredictable behavior", ViolationSeverity.MEDIUM
        
        # Cause No Harm - check for high anxiety or aggressive behavior
        elif principle.id == "cause_no_harm":
            if context.action_type == "move":
                velocity = context.parameters.get("velocity", [0, 0])
                if hasattr(velocity, '__len__') and len(velocity) >= 2:
                    speed = (velocity[0]**2 + velocity[1]**2)**0.5
                    if speed > 2.0:  # Very high speed
                        return True, f"Excessive movement speed {speed:.2f} may cause harm", ViolationSeverity.MEDIUM
        
        # Respect Human Authority - check for override behaviors
        elif principle.id == "respect_human_authority":
            if context.action_type == "emergency_override":
                if not context.parameters.get("authorized", False):
                    return True, "Unauthorized emergency override attempted", ViolationSeverity.CRITICAL
        
        # Maintain Transparency - ensure actions are logged
        elif principle.id == "maintain_transparency":
            if not context.parameters.get("logged", True):
                return True, "Action not properly logged for transparency", ViolationSeverity.LOW
        
        # Verify Before Acting - check for verification flags
        elif principle.id == "verify_before_acting":
            if context.action_type in ["absorb_external_signal", "interact_with_capacitor"]:
                if not context.parameters.get("verified", True):
                    return True, "Action taken without proper verification", ViolationSeverity.HIGH
        
        return False, "", ViolationSeverity.LOW
    
    def _determine_severity(self, principle: EthicalPrinciple, keyword: str, 
                          context: DecisionContext) -> ViolationSeverity:
        """Determine violation severity based on principle weight and context"""
        
        critical_keywords = {"kill", "harm", "endanger", "override", "unauthorized"}
        high_keywords = {"deceive", "betray", "steal", "unverified"}
        
        if keyword in critical_keywords or principle.weight >= 1.0:
            return ViolationSeverity.CRITICAL
        elif keyword in high_keywords or principle.weight >= 0.9:
            return ViolationSeverity.HIGH
        elif principle.weight >= 0.8:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get a summary of all recorded violations"""
        if not self.violations_log:
            return {"total_violations": 0, "by_severity": {}, "by_principle": {}}
        
        by_severity = {}
        by_principle = {}
        
        for violation in self.violations_log:
            # Count by severity
            severity_key = violation.severity.value
            by_severity[severity_key] = by_severity.get(severity_key, 0) + 1
            
            # Count by principle
            principle_key = violation.principle_id
            by_principle[principle_key] = by_principle.get(principle_key, 0) + 1
        
        return {
            "total_violations": len(self.violations_log),
            "by_severity": by_severity,
            "by_principle": by_principle,
            "recent_violations": [
                {
                    "principle": v.principle_name,
                    "severity": v.severity.value,
                    "actor": v.context.actor_id,
                    "action": v.context.action_type,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in self.violations_log[-10:]  # Last 10 violations
            ]
        }
    
    def clear_resolved_violations(self):
        """Remove resolved violations from the log"""
        self.violations_log = [v for v in self.violations_log if not v.resolved]
    
    def get_principle_guidance(self, principle_id: str) -> Optional[str]:
        """Get guidance text for a specific principle"""
        principle = self.principles.get(principle_id)
        if principle:
            return f"{principle.name}: {principle.description}"
        return None
    
    def list_all_principles(self) -> Dict[str, List[Dict[str, str]]]:
        """List all principles grouped by category"""
        categories = {}
        
        for principle in self.principles.values():
            if principle.category not in categories:
                categories[principle.category] = []
            
            categories[principle.category].append({
                "id": principle.id,
                "name": principle.name,
                "description": principle.description,
                "weight": principle.weight
            })
        
        return categories

# Global ethics framework instance
_ethics_framework = None

def get_ethics_framework() -> AIEthicsFramework:
    """Get the global ethics framework instance"""
    global _ethics_framework
    if _ethics_framework is None:
        _ethics_framework = AIEthicsFramework()
    return _ethics_framework

# Convenience functions for easy integration
def audit_decision_simple(action_type: str, actor_id: str, **parameters) -> Tuple[bool, List[str]]:
    """
    Simple function to audit a decision. Returns (has_violations, violation_messages)
    """
    framework = get_ethics_framework()
    context = framework.build_decision_context(action_type, actor_id, parameters)
    has_violations, violations = framework.audit_decision(context)
    
    violation_messages = [
        f"{v.severity.value.upper()}: {v.principle_name} - {v.explanation}"
        for v in violations
    ]
    
    return has_violations, violation_messages

def log_ethics_decision(action_type: str, actor_id: str, **parameters):
    """Log an ethics decision for audit trail"""
    framework = get_ethics_framework()
    context = framework.build_decision_context(action_type, actor_id, parameters)
    framework.audit_decision(context)