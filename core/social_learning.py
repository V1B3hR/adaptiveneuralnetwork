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
from typing import Any

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
    context: dict[str, Any]
    credibility: float  # How credible is the model
    attention_weight: float  # How much attention was paid
    timestamp: float


@dataclass
class ConsensusProposal:
    """A proposal for group consensus"""
    proposer_id: int
    content: Any
    support_votes: list[int]
    reject_votes: list[int]
    confidence_scores: dict[int, float]  # node_id -> confidence
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
            'model_attractiveness': 0.5,  # How attractive the model is
            'model_competence': 0.8,      # How competent the model appears
            'behavior_novelty': 0.3,      # How novel the behavior is
            'personal_relevance': 0.7     # How relevant to personal goals
        }

        # Retention system - memory of observed behaviors
        self.behavioral_memory = deque(maxlen=100)
        self.successful_patterns = {}  # behavior -> success_rate

        # Reproduction capabilities
        self.skill_confidence = defaultdict(float)  # behavior -> confidence in reproducing
        self.practice_history = defaultdict(list)   # behavior -> [attempts]

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

    def observe_behavior(self, model_agent_id: int, behavior: str,
                        outcome: float, context: dict[str, Any]) -> SocialLearningObservation:
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
            self.attention_factors['model_competence'] * model_competence +
            self.attention_factors['behavior_novelty'] * behavior_novelty +
            self.attention_factors['personal_relevance'] * personal_relevance
        ) / 3.0

        # Create observation
        observation = SocialLearningObservation(
            model_node_id=model_agent_id,
            behavior=behavior,
            outcome=outcome,
            context=context,
            credibility=model_trust,
            attention_weight=attention_weight,
            timestamp=time.time()
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
        similar_behaviors = [b for b in self.skill_confidence.keys()
                           if self._behaviors_similar(b, behavior)]
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

    def update_trust(self, agent_id: int, interaction_outcome: float,
                    behavior_accuracy: float = 1.0):
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
            timestamp=time.time()
        )

        self.active_proposals[proposal_id] = proposal
        return proposal

    def vote_on_proposal(self, proposal: ConsensusProposal,
                        support: bool, confidence: float) -> ConsensusProposal:
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

    def evaluate_consensus(self, proposal: ConsensusProposal,
                          total_agents: int) -> tuple[bool, float]:
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
            weighted_support = sum(
                proposal.confidence_scores.get(agent_id, 0.5)
                for agent_id in proposal.support_votes
            ) / total_confidence
        else:
            weighted_support = support_ratio

        # Consensus reached if weighted support exceeds threshold
        consensus_reached = weighted_support >= self.consensus_threshold

        return consensus_reached, weighted_support

    def handle_ambiguous_signal(self, signal: Any, context: dict[str, Any]) -> dict[str, Any]:
        """
        Handle ambiguous or conflicting signals from multiple sources.
        
        Uses trust scores and past experience to resolve ambiguity.
        """
        interpretations = []

        # Get different interpretations based on past experience
        if hasattr(signal, 'source_id'):
            source_trust = self.trust_scores.get(signal.source_id, 0.5)

            # Weight interpretation by source trust
            interpretation = {
                'content': signal,
                'confidence': source_trust,
                'source_trust': source_trust,
                'context_match': self._calculate_context_match(context)
            }
            interpretations.append(interpretation)

        # Select best interpretation
        if interpretations:
            best_interpretation = max(interpretations,
                                    key=lambda x: x['confidence'] * x['context_match'])
            return {
                'interpretation': best_interpretation['content'],
                'confidence': best_interpretation['confidence'],
                'ambiguity_resolved': True
            }
        else:
            return {
                'interpretation': signal,
                'confidence': 0.3,
                'ambiguity_resolved': False
            }

    def _behaviors_similar(self, behavior1: str, behavior2: str) -> bool:
        """Simple behavior similarity check."""
        # Basic string similarity - could be enhanced with semantic similarity
        return len(set(behavior1.split()) & set(behavior2.split())) > 0

    def _calculate_context_match(self, context: dict[str, Any]) -> float:
        """Calculate how well a context matches past successful contexts."""
        # Simple implementation - could be enhanced with more sophisticated matching
        if not hasattr(self, '_past_contexts'):
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

    def facilitate_observation(self, observer_id: int, model_id: int,
                             behavior: str, outcome: float, context: dict[str, Any]):
        """Facilitate observation between agents."""
        if 0 <= observer_id < len(self.agents) and 0 <= model_id < len(self.agents):
            observer = self.agents[observer_id]
            observation = observer.observe_behavior(model_id, behavior, outcome, context)

            # Record interaction
            self.interaction_history.append({
                'type': 'observation',
                'observer': observer_id,
                'model': model_id,
                'behavior': behavior,
                'outcome': outcome,
                'attention_weight': observation.attention_weight,
                'timestamp': time.time()
            })

            return observation
        return None

    def run_consensus_round(self, proposal: ConsensusProposal) -> dict[str, Any]:
        """Run a round of consensus building."""
        results = {
            'proposal_id': proposal.proposal_id,
            'votes_collected': 0,
            'consensus_reached': False,
            'final_status': 'pending'
        }

        # Let each agent vote
        for agent in self.agents:
            if agent.agent_id != proposal.proposer_id:  # Proposer doesn't vote on their own
                # Agent decides whether to support based on their assessment
                support_prob = random.random()  # Could be more sophisticated
                confidence = random.uniform(0.3, 0.9)

                agent.vote_on_proposal(proposal, support_prob > 0.5, confidence)
                results['votes_collected'] += 1

        # Check for consensus
        consensus_reached, consensus_strength = self.agents[0].evaluate_consensus(
            proposal, len(self.agents)
        )

        results['consensus_reached'] = consensus_reached
        results['consensus_strength'] = consensus_strength

        if consensus_reached:
            proposal.status = 'accepted'
            results['final_status'] = 'accepted'
        elif len(proposal.support_votes) + len(proposal.reject_votes) >= len(self.agents) - 1:
            # All agents have voted but no consensus
            proposal.status = 'rejected'
            results['final_status'] = 'rejected'

        return results

    def resolve_conflict(self, conflicting_proposals: list[ConsensusProposal]) -> ConsensusProposal | None:
        """Resolve conflicts between multiple proposals."""
        if not conflicting_proposals:
            return None

        # Score proposals based on support and confidence
        proposal_scores = []
        for proposal in conflicting_proposals:
            total_support = len(proposal.support_votes)
            avg_confidence = (sum(proposal.confidence_scores.values()) /
                            max(len(proposal.confidence_scores), 1))

            score = total_support * avg_confidence
            proposal_scores.append((proposal, score))

        # Return proposal with highest score
        best_proposal = max(proposal_scores, key=lambda x: x[1])[0]
        return best_proposal

    def get_learning_statistics(self) -> dict[str, Any]:
        """Get statistics about social learning in the environment."""
        stats = {
            'total_agents': len(self.agents),
            'total_interactions': len(self.interaction_history),
            'observation_patterns': defaultdict(int),
            'trust_network_density': 0.0,
            'consensus_success_rate': 0.0
        }

        # Analyze observation patterns
        for interaction in self.interaction_history:
            if interaction['type'] == 'observation':
                stats['observation_patterns'][interaction['behavior']] += 1

        # Calculate trust network density
        total_trust_connections = 0
        total_possible_connections = len(self.agents) * (len(self.agents) - 1)

        for agent in self.agents:
            total_trust_connections += len(agent.trust_scores)

        if total_possible_connections > 0:
            stats['trust_network_density'] = total_trust_connections / total_possible_connections

        return stats


class SwarmIntelligenceBehavior:
    """
    Swarm intelligence behaviors for multi-agent coordination.
    
    Implements particle swarm optimization-inspired collective behaviors,
    ant colony optimization patterns, and flocking behaviors.
    """

    def __init__(self, agent_id: int, swarm_size: int):
        self.agent_id = agent_id
        self.swarm_size = swarm_size

        # Swarm parameters
        self.position = np.random.randn(10)  # Agent's position in solution space
        self.velocity = np.zeros(10)
        self.personal_best = self.position.copy()
        self.personal_best_fitness = -np.inf

        # PSO parameters
        self.inertia = 0.7
        self.cognitive_weight = 1.4  # Personal best influence
        self.social_weight = 1.4     # Global best influence

        # Flocking parameters
        self.separation_radius = 2.0
        self.alignment_radius = 5.0
        self.cohesion_radius = 8.0

        # Pheromone trail (ant colony inspired)
        self.pheromone_strength = 1.0
        self.pheromone_decay = 0.1

    def update_pso_position(self, global_best_position: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Update position using Particle Swarm Optimization dynamics."""
        # PSO velocity update
        r1, r2 = np.random.random(2)

        cognitive_component = self.cognitive_weight * r1 * (self.personal_best - self.position)
        social_component = self.social_weight * r2 * (global_best_position - self.position)

        self.velocity = (self.inertia * self.velocity +
                        cognitive_component + social_component)

        # Update position
        self.position += self.velocity * dt

        return self.position

    def flocking_behavior(self, neighbor_positions: list[np.ndarray], neighbor_velocities: list[np.ndarray]) -> np.ndarray:
        """Implement flocking behavior (separation, alignment, cohesion)."""
        if not neighbor_positions:
            return np.zeros_like(self.velocity)

        neighbor_positions = np.array(neighbor_positions)
        neighbor_velocities = np.array(neighbor_velocities)

        # Calculate distances to neighbors
        distances = np.linalg.norm(neighbor_positions - self.position, axis=1)

        # Separation: avoid crowding local flockmates
        separation = np.zeros_like(self.position)
        close_neighbors = distances < self.separation_radius
        if np.any(close_neighbors):
            diff = self.position - neighbor_positions[close_neighbors]
            diff_norm = np.linalg.norm(diff, axis=1, keepdims=True)
            diff_norm[diff_norm == 0] = 1e-6  # Avoid division by zero
            separation = np.mean(diff / diff_norm, axis=0)

        # Alignment: steer towards average heading of neighbors
        alignment = np.zeros_like(self.velocity)
        align_neighbors = distances < self.alignment_radius
        if np.any(align_neighbors):
            alignment = np.mean(neighbor_velocities[align_neighbors], axis=0) - self.velocity

        # Cohesion: steer towards average position of neighbors
        cohesion = np.zeros_like(self.position)
        cohesion_neighbors = distances < self.cohesion_radius
        if np.any(cohesion_neighbors):
            center_of_mass = np.mean(neighbor_positions[cohesion_neighbors], axis=0)
            cohesion = center_of_mass - self.position

        # Combine behaviors
        flocking_force = separation * 1.5 + alignment * 1.0 + cohesion * 1.0
        return flocking_force

    def ant_colony_decision(self, options: list[dict[str, Any]], pheromone_trails: dict[str, float]) -> int:
        """Make decision based on ant colony optimization principles."""
        if not options:
            return -1

        # Calculate probabilities based on pheromone trails and heuristic information
        probabilities = []

        for i, option in enumerate(options):
            option_key = option.get('key', f'option_{i}')
            pheromone = pheromone_trails.get(option_key, 0.1)
            heuristic = option.get('quality', 0.5)  # Heuristic information

            # Probability combines pheromone trail strength and heuristic desirability
            prob = (pheromone ** self.cognitive_weight) * (heuristic ** self.social_weight)
            probabilities.append(prob)

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(options)] * len(options)

        # Select option based on probabilities
        return np.random.choice(len(options), p=probabilities)

    def deposit_pheromone(self, path: list[str], success: float) -> dict[str, float]:
        """Deposit pheromone on successful paths."""
        pheromone_deposit = {}
        deposit_amount = success * self.pheromone_strength

        for step in path:
            pheromone_deposit[step] = deposit_amount

        return pheromone_deposit


class NegotiationProtocol:
    """
    Advanced negotiation and consensus protocols for multi-agent coordination.
    
    Implements auction-based mechanisms, voting protocols, and 
    game-theory inspired negotiation strategies.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.negotiation_history = []
        self.reputation_scores = {}
        self.negotiation_strategies = ['cooperative', 'competitive', 'adaptive']
        self.current_strategy = 'adaptive'

        # Auction parameters
        self.bid_strategy = 'truthful'  # 'truthful', 'strategic', 'competitive'
        self.reservation_values = {}

        # Game theory parameters
        self.cooperation_tendency = 0.6
        self.risk_tolerance = 0.4

    def initiate_auction(self, item: dict[str, Any], auction_type: str = 'first_price') -> dict[str, Any]:
        """Initiate an auction for resource allocation."""
        auction = {
            'auction_id': f'auction_{self.agent_id}_{time.time()}',
            'auctioneer': self.agent_id,
            'item': item,
            'type': auction_type,  # 'first_price', 'second_price', 'english'
            'bids': [],
            'status': 'open',
            'start_time': time.time(),
            'duration': item.get('duration', 30.0)  # seconds
        }

        return auction

    def submit_bid(self, auction: dict[str, Any], private_value: float) -> dict[str, Any]:
        """Submit a bid in an auction based on strategy."""
        if auction['status'] != 'open':
            return None

        # Calculate bid based on strategy and auction type
        if self.bid_strategy == 'truthful':
            bid_amount = private_value
        elif self.bid_strategy == 'strategic':
            if auction['type'] == 'first_price':
                # Shade bid in first-price auction
                bid_amount = private_value * 0.8
            else:
                # Bid truthfully in second-price auction
                bid_amount = private_value
        else:  # competitive
            # Bid slightly above estimated competition
            estimated_competition = np.mean([bid['amount'] for bid in auction['bids']]) if auction['bids'] else private_value * 0.5
            bid_amount = max(private_value * 0.9, estimated_competition * 1.1)

        bid = {
            'bidder': self.agent_id,
            'amount': bid_amount,
            'timestamp': time.time(),
            'private_value': private_value  # Hidden from others
        }

        return bid

    def resolve_auction(self, auction: dict[str, Any]) -> dict[str, Any]:
        """Resolve auction and determine winner."""
        if not auction['bids']:
            return {'winner': None, 'price': 0, 'status': 'no_bids'}

        sorted_bids = sorted(auction['bids'], key=lambda x: x['amount'], reverse=True)

        if auction['type'] == 'first_price':
            winner = sorted_bids[0]
            price = winner['amount']
        elif auction['type'] == 'second_price':
            winner = sorted_bids[0]
            price = sorted_bids[1]['amount'] if len(sorted_bids) > 1 else winner['amount']
        else:  # english auction
            winner = sorted_bids[0]
            price = winner['amount']

        result = {
            'winner': winner['bidder'],
            'price': price,
            'status': 'completed',
            'efficiency': self._calculate_auction_efficiency(auction, winner)
        }

        # Update reputation based on successful participation
        self.reputation_scores[winner['bidder']] = self.reputation_scores.get(winner['bidder'], 0.5) + 0.1

        return result

    def multi_issue_negotiation(self, issues: list[dict[str, Any]], opponent_preferences: dict[str, float]) -> dict[str, Any]:
        """Conduct multi-issue negotiation using integrative bargaining."""

        # Analyze issue importance and find potential trade-offs
        my_priorities = self._analyze_issue_priorities(issues)

        # Generate multiple offers exploring the solution space
        offers = []
        for _ in range(5):  # Generate 5 different offers
            offer = {}
            for issue in issues:
                issue_name = issue['name']

                # Balance my priorities with estimated opponent preferences
                my_weight = my_priorities.get(issue_name, 0.5)
                opponent_weight = opponent_preferences.get(issue_name, 0.5)

                # Create integrative solution
                if my_weight > opponent_weight:
                    # I care more, claim more value on this issue
                    offer[issue_name] = issue['range'][1] * 0.8 + issue['range'][0] * 0.2
                else:
                    # Opponent cares more, concede on this issue
                    offer[issue_name] = issue['range'][0] * 0.8 + issue['range'][1] * 0.2

            offer_value = self._calculate_offer_value(offer, my_priorities)
            offers.append({'terms': offer, 'value': offer_value})

        # Select best offer
        best_offer = max(offers, key=lambda x: x['value'])

        negotiation_result = {
            'offer': best_offer['terms'],
            'expected_value': best_offer['value'],
            'strategy': 'integrative',
            'concessions': self._identify_concessions(best_offer['terms'], my_priorities)
        }

        self.negotiation_history.append(negotiation_result)
        return negotiation_result

    def _calculate_auction_efficiency(self, auction: dict[str, Any], winner: dict[str, Any]) -> float:
        """Calculate allocative efficiency of auction outcome."""
        # Efficiency = winner's value / highest possible value
        all_values = [bid['private_value'] for bid in auction['bids']]
        highest_value = max(all_values)
        winner_value = winner['private_value']

        return winner_value / highest_value if highest_value > 0 else 1.0

    def _analyze_issue_priorities(self, issues: list[dict[str, Any]]) -> dict[str, float]:
        """Analyze relative importance of negotiation issues."""
        priorities = {}
        total_weight = 0

        for issue in issues:
            # Simple heuristic: larger ranges indicate more important issues
            range_size = issue['range'][1] - issue['range'][0]
            importance = issue.get('importance', 1.0)
            weight = range_size * importance

            priorities[issue['name']] = weight
            total_weight += weight

        # Normalize priorities
        if total_weight > 0:
            priorities = {k: v / total_weight for k, v in priorities.items()}

        return priorities

    def _calculate_offer_value(self, offer: dict[str, Any], priorities: dict[str, float]) -> float:
        """Calculate subjective value of an offer."""
        total_value = 0
        for issue, value in offer.items():
            weight = priorities.get(issue, 1.0)
            total_value += weight * value

        return total_value

    def _identify_concessions(self, offer: dict[str, Any], priorities: dict[str, float]) -> list[str]:
        """Identify concessions made in an offer."""
        concessions = []
        for issue, value in offer.items():
            priority = priorities.get(issue, 0.5)
            if priority > 0.7 and value < 0.5:  # High priority but low value
                concessions.append(issue)

        return concessions


class CompetitiveCooperativeEnvironment:
    """
    Environment supporting both competitive and cooperative multi-agent interactions.
    
    Manages resource allocation, territory control, coalition formation,
    and mixed-motive games.
    """

    def __init__(self, num_agents: int, environment_type: str = 'mixed'):
        self.num_agents = num_agents
        self.environment_type = environment_type  # 'competitive', 'cooperative', 'mixed'
        self.agents = []

        # Environment state
        self.resources = {'food': 100, 'territory': 50, 'information': 75}
        self.territories = {}  # agent_id -> territory_size
        self.coalitions = []   # List of agent coalitions

        # Game mechanics
        self.round_number = 0
        self.resource_regeneration_rate = 0.1
        self.competition_intensity = 0.5

        # Performance tracking
        self.agent_scores = defaultdict(float)
        self.interaction_history = []

    def add_agent(self, agent_id: int, initial_resources: dict[str, float] | None = None) -> None:
        """Add an agent to the environment."""
        if initial_resources is None:
            initial_resources = {'food': 10, 'territory': 5, 'information': 8}

        agent_data = {
            'id': agent_id,
            'resources': initial_resources.copy(),
            'strategies': [],
            'partnerships': set(),
            'reputation': 0.5
        }

        self.agents.append(agent_data)
        self.territories[agent_id] = initial_resources.get('territory', 5)

    def run_competitive_round(self) -> dict[str, Any]:
        """Run a round of competitive interactions."""
        round_results = {
            'round': self.round_number,
            'type': 'competitive',
            'interactions': [],
            'resource_changes': {}
        }

        # Random pairwise competitions
        available_agents = list(range(len(self.agents)))

        while len(available_agents) >= 2:
            # Select two agents for competition
            agent1_idx = available_agents.pop(random.randint(0, len(available_agents) - 1))
            agent2_idx = available_agents.pop(random.randint(0, len(available_agents) - 1))

            agent1 = self.agents[agent1_idx]
            agent2 = self.agents[agent2_idx]

            # Resource competition
            resource_type = random.choice(['food', 'territory', 'information'])
            competition_result = self._compete_for_resource(agent1, agent2, resource_type)

            round_results['interactions'].append(competition_result)

            # Update scores
            self.agent_scores[agent1['id']] += competition_result['agent1_gain']
            self.agent_scores[agent2['id']] += competition_result['agent2_gain']

        self.round_number += 1
        return round_results

    def run_cooperative_round(self) -> dict[str, Any]:
        """Run a round of cooperative interactions."""
        round_results = {
            'round': self.round_number,
            'type': 'cooperative',
            'coalitions_formed': [],
            'collective_gains': {}
        }

        # Attempt coalition formation
        potential_coalitions = self._identify_beneficial_coalitions()

        for coalition in potential_coalitions:
            if self._form_coalition(coalition):
                # Execute cooperative task
                task_result = self._execute_cooperative_task(coalition)
                round_results['coalitions_formed'].append(task_result)

                # Distribute rewards
                total_reward = task_result['collective_reward']
                reward_per_agent = total_reward / len(coalition)

                for agent_id in coalition:
                    self.agent_scores[agent_id] += reward_per_agent

        self.round_number += 1
        return round_results

    def run_mixed_round(self) -> dict[str, Any]:
        """Run a round with both competitive and cooperative elements."""
        # Randomly decide on competition vs cooperation
        if random.random() < self.competition_intensity:
            return self.run_competitive_round()
        else:
            return self.run_cooperative_round()

    def _compete_for_resource(self, agent1: dict, agent2: dict, resource_type: str) -> dict[str, Any]:
        """Simulate competition between two agents for a resource."""

        # Competition strength based on current resources and strategy
        strength1 = agent1['resources'].get(resource_type, 0) + random.random()
        strength2 = agent2['resources'].get(resource_type, 0) + random.random()

        # Determine winner and resource transfer
        if strength1 > strength2:
            winner, loser = agent1, agent2
            transfer_amount = min(loser['resources'].get(resource_type, 0) * 0.3, 5)
        else:
            winner, loser = agent2, agent1
            transfer_amount = min(loser['resources'].get(resource_type, 0) * 0.3, 5)

        # Execute transfer
        loser['resources'][resource_type] = max(0, loser['resources'][resource_type] - transfer_amount)
        winner['resources'][resource_type] += transfer_amount * 0.8  # Some loss in transfer

        return {
            'type': 'competition',
            'resource': resource_type,
            'winner': winner['id'],
            'loser': loser['id'],
            'transfer_amount': transfer_amount,
            'agent1_gain': transfer_amount * 0.8 if winner == agent1 else -transfer_amount,
            'agent2_gain': transfer_amount * 0.8 if winner == agent2 else -transfer_amount
        }

    def _identify_beneficial_coalitions(self) -> list[list[int]]:
        """Identify potentially beneficial coalitions."""
        coalitions = []

        # Find agents with complementary resources
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent1, agent2 = self.agents[i], self.agents[j]

                # Check resource complementarity
                synergy_score = 0
                for resource in ['food', 'territory', 'information']:
                    resource1 = agent1['resources'].get(resource, 0)
                    resource2 = agent2['resources'].get(resource, 0)

                    # High complementarity if one is strong where other is weak
                    if (resource1 > 15 and resource2 < 5) or (resource2 > 15 and resource1 < 5):
                        synergy_score += 1

                if synergy_score >= 1:  # At least one complementary resource
                    coalitions.append([agent1['id'], agent2['id']])

        return coalitions

    def _form_coalition(self, coalition: list[int]) -> bool:
        """Attempt to form a coalition between agents."""
        # All agents must agree to join
        for agent_id in coalition:
            agent = next(a for a in self.agents if a['id'] == agent_id)

            # Simple decision: join if potential partners have good reputation
            partners = [aid for aid in coalition if aid != agent_id]
            avg_partner_reputation = np.mean([
                next(a for a in self.agents if a['id'] == pid)['reputation']
                for pid in partners
            ])

            if avg_partner_reputation < 0.3:  # Don't join with untrustworthy agents
                return False

        # Coalition formed successfully
        self.coalitions.append({
            'members': coalition,
            'formed_round': self.round_number,
            'trust_level': avg_partner_reputation
        })

        return True

    def _execute_cooperative_task(self, coalition: list[int]) -> dict[str, Any]:
        """Execute a cooperative task for the coalition."""

        # Calculate collective capability
        total_resources = defaultdict(float)
        for agent_id in coalition:
            agent = next(a for a in self.agents if a['id'] == agent_id)
            for resource, amount in agent['resources'].items():
                total_resources[resource] += amount

        # Task difficulty and reward scale with coalition size
        base_reward = len(coalition) * 10
        synergy_bonus = (len(coalition) - 1) * 5  # Cooperation bonus

        # Success probability based on resource adequacy
        required_resources = {'food': len(coalition) * 8, 'territory': len(coalition) * 4, 'information': len(coalition) * 6}
        success_prob = 1.0

        for resource, required in required_resources.items():
            if total_resources[resource] < required:
                success_prob *= total_resources[resource] / required

        success = random.random() < success_prob
        collective_reward = (base_reward + synergy_bonus) * success_prob

        # Update agent reputations based on contribution and success
        for agent_id in coalition:
            agent = next(a for a in self.agents if a['id'] == agent_id)
            if success:
                agent['reputation'] = min(1.0, agent['reputation'] + 0.1)
            else:
                agent['reputation'] = max(0.0, agent['reputation'] - 0.05)

        return {
            'coalition': coalition,
            'task_type': 'resource_gathering',
            'success': success,
            'collective_reward': collective_reward,
            'success_probability': success_prob
        }

    def get_environment_state(self) -> dict[str, Any]:
        """Get current state of the environment."""
        return {
            'round': self.round_number,
            'agents': len(self.agents),
            'resources': self.resources.copy(),
            'active_coalitions': len(self.coalitions),
            'average_score': np.mean(list(self.agent_scores.values())) if self.agent_scores else 0,
            'competition_level': self.competition_intensity,
            'cooperation_events': len([c for c in self.coalitions if c['formed_round'] >= self.round_number - 5])
        }
