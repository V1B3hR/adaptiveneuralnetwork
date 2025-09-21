"""
Social Learning System based on Bandura's Social Learning Theory

This module implements multi-agent social learning, consensus building,
and communication protocols as required by Phase 2.2.

Reference: https://www.simplypsychology.org/bandura.html
Key concepts: observational learning, modeling, imitation, social reinforcement
"""

import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LearningProcess(Enum):
    """Bandura's four processes of observational learning"""

    ATTENTION = "attention"
    RETENTION = "retention"
    REPRODUCTION = "reproduction"
    MOTIVATION = "motivation"


@dataclass
class SocialLearningObservation:
    """Represents an observation for social learning"""

    model_node_id: int
    behavior: str
    outcome: float  # Success/reward of the behavior
    context: Dict[str, Any]
    credibility: float  # How credible is the model
    attention_weight: float  # How much attention was paid
    timestamp: float


@dataclass
class ConsensusProposal:
    """A proposal for group consensus"""

    proposer_id: int
    content: Any
    support_votes: List[int]
    reject_votes: List[int]
    confidence_scores: Dict[int, float]  # node_id -> confidence
    proposal_id: str
    timestamp: float
    status: str = "pending"  # pending, accepted, rejected


class SocialLearningAgent:
    """
    Agent capable of social learning based on Bandura's theory.

    Implements the four key processes:
    1. Attention - paying attention to models
    2. Retention - remembering what was observed
    3. Reproduction - being able to reproduce the behavior
    4. Motivation - having incentive to perform the behavior
    """

    def __init__(self, agent_id: int, learning_rate: float = 0.1):
        self.agent_id = agent_id
        self.learning_rate = learning_rate

        # Bandura's four processes
        self.attention_factors = {
            "model_attractiveness": 0.5,  # How attractive the model is
            "model_competence": 0.8,  # How competent the model appears
            "behavior_novelty": 0.3,  # How novel the behavior is
            "personal_relevance": 0.7,  # How relevant to personal goals
        }

        # Retention system - memory of observed behaviors
        self.behavioral_memory = deque(maxlen=100)
        self.successful_patterns = {}  # behavior -> success_rate

        # Reproduction capabilities
        self.skill_confidence = defaultdict(float)  # behavior -> confidence in reproducing
        self.practice_history = defaultdict(list)  # behavior -> [attempts]

        # Motivation factors
        self.intrinsic_motivation = 0.6
        self.extrinsic_rewards = {}  # behavior -> expected_reward
        self.social_approval_seeking = 0.4

        # Trust and influence networks
        self.trust_scores = {}  # agent_id -> trust_score
        self.influence_susceptibility = {}  # agent_id -> susceptibility

        # Consensus building
        self.active_proposals = {}  # proposal_id -> ConsensusProposal
        self.consensus_threshold = 0.7

    def observe_behavior(
        self, model_agent_id: int, behavior: str, outcome: float, context: Dict[str, Any]
    ) -> SocialLearningObservation:
        """
        Observe a behavior from another agent (Attention process).

        Following Bandura's attention process, determines how much attention
        to pay based on model characteristics and behavior properties.
        """
        # Calculate attention weight based on multiple factors
        model_trust = self.trust_scores.get(model_agent_id, 0.5)
        model_competence = min(1.0, model_trust * 1.2)  # Trust influences perceived competence

        # Behavior novelty - new behaviors get more attention
        behavior_novelty = 1.0 if behavior not in self.successful_patterns else 0.3

        # Personal relevance - outcomes similar to our goals get more attention
        personal_relevance = min(1.0, abs(outcome) * 0.8)  # Higher outcomes are more relevant

        # Calculate overall attention weight
        attention_weight = (
            self.attention_factors["model_competence"] * model_competence
            + self.attention_factors["behavior_novelty"] * behavior_novelty
            + self.attention_factors["personal_relevance"] * personal_relevance
        ) / 3.0

        # Create observation
        observation = SocialLearningObservation(
            model_node_id=model_agent_id,
            behavior=behavior,
            outcome=outcome,
            context=context,
            credibility=model_trust,
            attention_weight=attention_weight,
            timestamp=time.time(),
        )

        # Store in memory if attention weight is sufficient (Retention process)
        if attention_weight > 0.3:
            self.retain_observation(observation)

        return observation

    def retain_observation(self, observation: SocialLearningObservation):
        """
        Retain an observation in memory (Retention process).

        Stores behavioral patterns and updates success expectations.
        """
        self.behavioral_memory.append(observation)

        # Update success patterns
        behavior = observation.behavior
        if behavior not in self.successful_patterns:
            self.successful_patterns[behavior] = []

        self.successful_patterns[behavior].append(observation.outcome)

        # Keep only recent outcomes for adaptation
        if len(self.successful_patterns[behavior]) > 10:
            self.successful_patterns[behavior] = self.successful_patterns[behavior][-10:]

    def can_reproduce_behavior(self, behavior: str) -> float:
        """
        Assess ability to reproduce a behavior (Reproduction process).

        Returns confidence score (0-1) of being able to reproduce the behavior.
        """
        # Base confidence from previous practice
        base_confidence = self.skill_confidence.get(behavior, 0.1)

        # Increase confidence if we've observed it multiple times
        observation_count = sum(1 for obs in self.behavioral_memory if obs.behavior == behavior)
        observation_bonus = min(0.3, observation_count * 0.05)

        # Consider similarity to known behaviors
        similar_behaviors = [
            b for b in self.skill_confidence.keys() if self._behaviors_similar(b, behavior)
        ]
        similarity_bonus = min(0.2, len(similar_behaviors) * 0.1)

        return min(1.0, base_confidence + observation_bonus + similarity_bonus)

    def get_motivation_to_perform(self, behavior: str) -> float:
        """
        Calculate motivation to perform a behavior (Motivation process).

        Combines intrinsic motivation, expected rewards, and social factors.
        """
        # Expected reward from observation
        if behavior in self.successful_patterns:
            expected_reward = np.mean(self.successful_patterns[behavior])
            reward_motivation = max(0, expected_reward * 0.3)
        else:
            reward_motivation = 0.1  # Small base motivation for exploration

        # Social approval factor
        social_motivation = self.social_approval_seeking * 0.2

        # Intrinsic motivation
        intrinsic = self.intrinsic_motivation * 0.3

        # Combine factors
        total_motivation = reward_motivation + social_motivation + intrinsic
        return min(1.0, total_motivation)

    def decide_to_imitate(self, behavior: str) -> bool:
        """
        Decide whether to imitate a behavior based on all four processes.

        This integrates Bandura's four processes to make a final decision.
        """
        # Must have observed the behavior (attention/retention)
        if behavior not in self.successful_patterns:
            return False

        # Check reproduction capability
        reproduction_confidence = self.can_reproduce_behavior(behavior)
        if reproduction_confidence < 0.3:
            return False

        # Check motivation
        motivation = self.get_motivation_to_perform(behavior)
        if motivation < 0.4:
            return False

        # Probabilistic decision based on all factors
        decision_probability = (reproduction_confidence + motivation) / 2.0
        return random.random() < decision_probability

    def update_trust(
        self, agent_id: int, interaction_outcome: float, behavior_accuracy: float = 1.0
    ):
        """Update trust score for another agent based on interaction outcomes."""
        current_trust = self.trust_scores.get(agent_id, 0.5)

        # Trust update based on outcome and accuracy
        outcome_factor = max(-0.3, min(0.3, interaction_outcome * 0.1))
        accuracy_factor = (behavior_accuracy - 0.5) * 0.2

        # Gradual trust update
        new_trust = current_trust + self.learning_rate * (outcome_factor + accuracy_factor)
        self.trust_scores[agent_id] = max(0.0, min(1.0, new_trust))

    def propose_consensus(self, content: Any, confidence: float) -> ConsensusProposal:
        """Propose something for group consensus."""
        proposal_id = f"prop_{self.agent_id}_{time.time()}"

        proposal = ConsensusProposal(
            proposer_id=self.agent_id,
            content=content,
            support_votes=[],
            reject_votes=[],
            confidence_scores={self.agent_id: confidence},
            proposal_id=proposal_id,
            timestamp=time.time(),
        )

        self.active_proposals[proposal_id] = proposal
        return proposal

    def vote_on_proposal(
        self, proposal: ConsensusProposal, support: bool, confidence: float
    ) -> ConsensusProposal:
        """Vote on a consensus proposal."""
        if support:
            if self.agent_id not in proposal.support_votes:
                proposal.support_votes.append(self.agent_id)
            # Remove from reject votes if was there
            if self.agent_id in proposal.reject_votes:
                proposal.reject_votes.remove(self.agent_id)
        else:
            if self.agent_id not in proposal.reject_votes:
                proposal.reject_votes.append(self.agent_id)
            # Remove from support votes if was there
            if self.agent_id in proposal.support_votes:
                proposal.support_votes.remove(self.agent_id)

        # Update confidence
        proposal.confidence_scores[self.agent_id] = confidence

        return proposal

    def evaluate_consensus(
        self, proposal: ConsensusProposal, total_agents: int
    ) -> Tuple[bool, float]:
        """
        Evaluate if consensus has been reached on a proposal.

        Returns (consensus_reached, consensus_strength)
        """
        total_votes = len(proposal.support_votes) + len(proposal.reject_votes)
        if total_votes == 0:
            return False, 0.0

        # Calculate support ratio
        support_ratio = len(proposal.support_votes) / total_votes

        # Weight by confidence scores
        total_confidence = sum(proposal.confidence_scores.values())
        if total_confidence > 0:
            weighted_support = (
                sum(
                    proposal.confidence_scores.get(agent_id, 0.5)
                    for agent_id in proposal.support_votes
                )
                / total_confidence
            )
        else:
            weighted_support = support_ratio

        # Consensus reached if weighted support exceeds threshold
        consensus_reached = weighted_support >= self.consensus_threshold

        return consensus_reached, weighted_support

    def handle_ambiguous_signal(self, signal: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle ambiguous or conflicting signals from multiple sources.

        Uses trust scores and past experience to resolve ambiguity.
        """
        interpretations = []

        # Get different interpretations based on past experience
        if hasattr(signal, "source_id"):
            source_trust = self.trust_scores.get(signal.source_id, 0.5)

            # Weight interpretation by source trust
            interpretation = {
                "content": signal,
                "confidence": source_trust,
                "source_trust": source_trust,
                "context_match": self._calculate_context_match(context),
            }
            interpretations.append(interpretation)

        # Select best interpretation
        if interpretations:
            best_interpretation = max(
                interpretations, key=lambda x: x["confidence"] * x["context_match"]
            )
            return {
                "interpretation": best_interpretation["content"],
                "confidence": best_interpretation["confidence"],
                "ambiguity_resolved": True,
            }
        else:
            return {"interpretation": signal, "confidence": 0.3, "ambiguity_resolved": False}

    def _behaviors_similar(self, behavior1: str, behavior2: str) -> bool:
        """Simple behavior similarity check."""
        # Basic string similarity - could be enhanced with semantic similarity
        return len(set(behavior1.split()) & set(behavior2.split())) > 0

    def _calculate_context_match(self, context: Dict[str, Any]) -> float:
        """Calculate how well a context matches past successful contexts."""
        # Simple implementation - could be enhanced with more sophisticated matching
        if not hasattr(self, "_past_contexts"):
            self._past_contexts = []
            return 0.5

        # Find similar past contexts
        similarities = []
        for past_context in self._past_contexts:
            similarity = len(set(context.keys()) & set(past_context.keys()))
            similarities.append(similarity)

        return max(similarities) / max(len(context), 1) if similarities else 0.5


class MultiAgentSocialLearningEnvironment:
    """Environment for multi-agent social learning and consensus building."""

    def __init__(self, num_agents: int):
        self.agents = [SocialLearningAgent(i) for i in range(num_agents)]
        self.global_proposals = {}
        self.interaction_history = []

    def facilitate_observation(
        self,
        observer_id: int,
        model_id: int,
        behavior: str,
        outcome: float,
        context: Dict[str, Any],
    ):
        """Facilitate observation between agents."""
        if 0 <= observer_id < len(self.agents) and 0 <= model_id < len(self.agents):
            observer = self.agents[observer_id]
            observation = observer.observe_behavior(model_id, behavior, outcome, context)

            # Record interaction
            self.interaction_history.append(
                {
                    "type": "observation",
                    "observer": observer_id,
                    "model": model_id,
                    "behavior": behavior,
                    "outcome": outcome,
                    "attention_weight": observation.attention_weight,
                    "timestamp": time.time(),
                }
            )

            return observation
        return None

    def run_consensus_round(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """Run a round of consensus building."""
        results = {
            "proposal_id": proposal.proposal_id,
            "votes_collected": 0,
            "consensus_reached": False,
            "final_status": "pending",
        }

        # Let each agent vote
        for agent in self.agents:
            if agent.agent_id != proposal.proposer_id:  # Proposer doesn't vote on their own
                # Agent decides whether to support based on their assessment
                support_prob = random.random()  # Could be more sophisticated
                confidence = random.uniform(0.3, 0.9)

                agent.vote_on_proposal(proposal, support_prob > 0.5, confidence)
                results["votes_collected"] += 1

        # Check for consensus
        consensus_reached, consensus_strength = self.agents[0].evaluate_consensus(
            proposal, len(self.agents)
        )

        results["consensus_reached"] = consensus_reached
        results["consensus_strength"] = consensus_strength

        if consensus_reached:
            proposal.status = "accepted"
            results["final_status"] = "accepted"
        elif len(proposal.support_votes) + len(proposal.reject_votes) >= len(self.agents) - 1:
            # All agents have voted but no consensus
            proposal.status = "rejected"
            results["final_status"] = "rejected"

        return results

    def resolve_conflict(
        self, conflicting_proposals: List[ConsensusProposal]
    ) -> Optional[ConsensusProposal]:
        """Resolve conflicts between multiple proposals."""
        if not conflicting_proposals:
            return None

        # Score proposals based on support and confidence
        proposal_scores = []
        for proposal in conflicting_proposals:
            total_support = len(proposal.support_votes)
            avg_confidence = sum(proposal.confidence_scores.values()) / max(
                len(proposal.confidence_scores), 1
            )

            score = total_support * avg_confidence
            proposal_scores.append((proposal, score))

        # Return proposal with highest score
        best_proposal = max(proposal_scores, key=lambda x: x[1])[0]
        return best_proposal

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about social learning in the environment."""
        stats = {
            "total_agents": len(self.agents),
            "total_interactions": len(self.interaction_history),
            "observation_patterns": defaultdict(int),
            "trust_network_density": 0.0,
            "consensus_success_rate": 0.0,
        }

        # Analyze observation patterns
        for interaction in self.interaction_history:
            if interaction["type"] == "observation":
                stats["observation_patterns"][interaction["behavior"]] += 1

        # Calculate trust network density
        total_trust_connections = 0
        total_possible_connections = len(self.agents) * (len(self.agents) - 1)

        for agent in self.agents:
            total_trust_connections += len(agent.trust_scores)

        if total_possible_connections > 0:
            stats["trust_network_density"] = total_trust_connections / total_possible_connections

        return stats
