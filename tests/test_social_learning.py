"""
Test for Social Learning System - Phase 2.2

Tests the multi-agent social learning system based on Bandura's social learning theory,
including consensus building and communication protocols.
"""

import unittest
import time
import numpy as np
from unittest.mock import Mock, patch

# Import from the core package
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.social_learning import (
    SocialLearningAgent, MultiAgentSocialLearningEnvironment,
    ConsensusProposal, SocialLearningObservation, LearningProcess
)


class TestSocialLearningAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment for social learning tests."""
        self.agent = SocialLearningAgent(agent_id=1, learning_rate=0.1)
        
    def test_agent_initialization(self):
        """Test that social learning agent initializes correctly."""
        self.assertEqual(self.agent.agent_id, 1)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertIn('model_competence', self.agent.attention_factors)
        self.assertEqual(len(self.agent.behavioral_memory), 0)
        
    def test_observe_behavior_attention_process(self):
        """Test Bandura's attention process in behavior observation."""
        # Set up trust for model agent
        model_agent_id = 2
        self.agent.trust_scores[model_agent_id] = 0.8
        
        # Observe a behavior
        observation = self.agent.observe_behavior(
            model_agent_id=model_agent_id,
            behavior="collaborative_problem_solving",
            outcome=0.9,  # High success
            context={"task": "path_finding", "difficulty": "high"}
        )
        
        # Verify observation properties
        self.assertIsInstance(observation, SocialLearningObservation)
        self.assertEqual(observation.model_node_id, model_agent_id)
        self.assertEqual(observation.behavior, "collaborative_problem_solving")
        self.assertEqual(observation.outcome, 0.9)
        self.assertGreater(observation.attention_weight, 0.0)
        self.assertLessEqual(observation.attention_weight, 1.0)
        
    def test_retention_process(self):
        """Test Bandura's retention process."""
        # Create observation
        observation = SocialLearningObservation(
            model_node_id=2,
            behavior="negotiation",
            outcome=0.7,
            context={"type": "resource_sharing"},
            credibility=0.8,
            attention_weight=0.6,
            timestamp=time.time()
        )
        
        # Retain observation
        self.agent.retain_observation(observation)
        
        # Verify retention
        self.assertEqual(len(self.agent.behavioral_memory), 1)
        self.assertIn("negotiation", self.agent.successful_patterns)
        self.assertEqual(self.agent.successful_patterns["negotiation"], [0.7])
        
    def test_reproduction_capability_assessment(self):
        """Test Bandura's reproduction process."""
        behavior = "cooperation_strategy"
        
        # Initially low confidence
        initial_confidence = self.agent.can_reproduce_behavior(behavior)
        self.assertLess(initial_confidence, 0.5)
        
        # Increase confidence through practice
        self.agent.skill_confidence[behavior] = 0.7
        improved_confidence = self.agent.can_reproduce_behavior(behavior)
        self.assertGreater(improved_confidence, initial_confidence)
        
    def test_motivation_calculation(self):
        """Test Bandura's motivation process."""
        behavior = "consensus_building"
        
        # Add successful pattern
        self.agent.successful_patterns[behavior] = [0.8, 0.9, 0.7]
        
        # Calculate motivation
        motivation = self.agent.get_motivation_to_perform(behavior)
        
        # Should be positive due to successful pattern
        self.assertGreater(motivation, 0.0)
        self.assertLessEqual(motivation, 1.0)
        
    def test_imitation_decision_integration(self):
        """Test integration of all four Bandura processes in imitation decision."""
        behavior = "conflict_resolution"
        
        # Set up successful observation and retention
        self.agent.successful_patterns[behavior] = [0.8, 0.9]
        
        # Set up reproduction confidence
        self.agent.skill_confidence[behavior] = 0.6
        
        # Set up motivation factors
        self.agent.intrinsic_motivation = 0.7
        
        # Test decision
        decision = self.agent.decide_to_imitate(behavior)
        self.assertIsInstance(decision, (bool, np.bool_))
        
        # With good setup, should be likely to imitate
        # Run multiple times to test probabilistic nature
        decisions = [self.agent.decide_to_imitate(behavior) for _ in range(10)]
        self.assertTrue(any(decisions), "Should make positive imitation decisions sometimes")
        
    def test_trust_updating(self):
        """Test trust score updating based on interactions."""
        agent_id = 3
        initial_trust = 0.5
        self.agent.trust_scores[agent_id] = initial_trust
        
        # Positive interaction
        self.agent.update_trust(agent_id, interaction_outcome=0.8, behavior_accuracy=0.9)
        new_trust = self.agent.trust_scores[agent_id]
        
        self.assertGreater(new_trust, initial_trust)
        self.assertLessEqual(new_trust, 1.0)
        
        # Negative interaction
        self.agent.update_trust(agent_id, interaction_outcome=-0.5, behavior_accuracy=0.3)
        updated_trust = self.agent.trust_scores[agent_id]
        
        self.assertLess(updated_trust, new_trust)
        self.assertGreaterEqual(updated_trust, 0.0)
        
    def test_consensus_proposal_creation(self):
        """Test creation of consensus proposals."""
        content = {"action": "explore_north", "priority": "high"}
        confidence = 0.8
        
        proposal = self.agent.propose_consensus(content, confidence)
        
        self.assertIsInstance(proposal, ConsensusProposal)
        self.assertEqual(proposal.proposer_id, self.agent.agent_id)
        self.assertEqual(proposal.content, content)
        self.assertEqual(proposal.confidence_scores[self.agent.agent_id], confidence)
        self.assertEqual(proposal.status, "pending")
        
    def test_voting_on_proposal(self):
        """Test voting mechanism for consensus proposals."""
        # Create proposal
        proposal = ConsensusProposal(
            proposer_id=2,
            content={"strategy": "defensive"},
            support_votes=[],
            reject_votes=[],
            confidence_scores={},
            proposal_id="test_prop_1",
            timestamp=time.time()
        )
        
        # Vote in support
        updated_proposal = self.agent.vote_on_proposal(proposal, support=True, confidence=0.7)
        
        self.assertIn(self.agent.agent_id, updated_proposal.support_votes)
        self.assertEqual(updated_proposal.confidence_scores[self.agent.agent_id], 0.7)
        
        # Change vote to reject
        updated_proposal = self.agent.vote_on_proposal(proposal, support=False, confidence=0.6)
        
        self.assertNotIn(self.agent.agent_id, updated_proposal.support_votes)
        self.assertIn(self.agent.agent_id, updated_proposal.reject_votes)
        self.assertEqual(updated_proposal.confidence_scores[self.agent.agent_id], 0.6)
        
    def test_consensus_evaluation(self):
        """Test consensus evaluation mechanism."""
        proposal = ConsensusProposal(
            proposer_id=2,
            content={"action": "retreat"},
            support_votes=[1, 3, 4],  # 3 support votes
            reject_votes=[5],          # 1 reject vote
            confidence_scores={1: 0.8, 3: 0.9, 4: 0.7, 5: 0.6},
            proposal_id="test_consensus",
            timestamp=time.time()
        )
        
        consensus_reached, consensus_strength = self.agent.evaluate_consensus(proposal, total_agents=5)
        
        self.assertTrue(consensus_reached)  # Should reach consensus with 3/4 votes and high confidence
        self.assertGreater(consensus_strength, 0.7)  # Should have strong consensus
        
    def test_ambiguous_signal_handling(self):
        """Test handling of ambiguous signals."""
        # Create mock signal with source
        signal = Mock()
        signal.source_id = 4
        signal.content = "ambiguous_instruction"
        
        # Set trust for source
        self.agent.trust_scores[4] = 0.8
        
        context = {"urgency": "high", "clarity": "low"}
        
        result = self.agent.handle_ambiguous_signal(signal, context)
        
        self.assertIn('interpretation', result)
        self.assertIn('confidence', result)
        self.assertIn('ambiguity_resolved', result)
        self.assertIsInstance(result['confidence'], float)
        

class TestMultiAgentSocialLearningEnvironment(unittest.TestCase):
    
    def setUp(self):
        """Set up multi-agent environment for testing."""
        self.env = MultiAgentSocialLearningEnvironment(num_agents=4)
        
    def test_environment_initialization(self):
        """Test multi-agent environment initialization."""
        self.assertEqual(len(self.env.agents), 4)
        self.assertEqual(len(self.env.global_proposals), 0)
        self.assertEqual(len(self.env.interaction_history), 0)
        
        # Check that agents have unique IDs
        agent_ids = [agent.agent_id for agent in self.env.agents]
        self.assertEqual(len(set(agent_ids)), 4)
        
    def test_facilitate_observation(self):
        """Test facilitation of observation between agents."""
        observer_id = 0
        model_id = 1
        behavior = "resource_sharing"
        outcome = 0.8
        context = {"resource_type": "energy", "scarcity": "medium"}
        
        observation = self.env.facilitate_observation(
            observer_id, model_id, behavior, outcome, context
        )
        
        self.assertIsInstance(observation, SocialLearningObservation)
        self.assertEqual(len(self.env.interaction_history), 1)
        
        interaction = self.env.interaction_history[0]
        self.assertEqual(interaction['type'], 'observation')
        self.assertEqual(interaction['observer'], observer_id)
        self.assertEqual(interaction['model'], model_id)
        self.assertEqual(interaction['behavior'], behavior)
        
    def test_consensus_round(self):
        """Test running a consensus round."""
        # Create proposal
        proposal = ConsensusProposal(
            proposer_id=0,
            content={"action": "form_alliance", "with": [1, 2]},
            support_votes=[],
            reject_votes=[],
            confidence_scores={0: 0.9},
            proposal_id="alliance_proposal",
            timestamp=time.time()
        )
        
        # Run consensus round
        results = self.env.run_consensus_round(proposal)
        
        self.assertIn('proposal_id', results)
        self.assertIn('votes_collected', results)
        self.assertIn('consensus_reached', results)
        self.assertIn('final_status', results)
        
        # Should have collected votes from 3 agents (excluding proposer)
        self.assertEqual(results['votes_collected'], 3)
        
    def test_conflict_resolution(self):
        """Test conflict resolution between competing proposals."""
        # Create competing proposals
        proposal1 = ConsensusProposal(
            proposer_id=0,
            content={"strategy": "aggressive"},
            support_votes=[1, 2],
            reject_votes=[],
            confidence_scores={0: 0.8, 1: 0.7, 2: 0.9},
            proposal_id="aggressive_strategy",
            timestamp=time.time()
        )
        
        proposal2 = ConsensusProposal(
            proposer_id=3,
            content={"strategy": "defensive"},
            support_votes=[1],
            reject_votes=[],
            confidence_scores={3: 0.9, 1: 0.6},
            proposal_id="defensive_strategy",
            timestamp=time.time()
        )
        
        # Resolve conflict
        winning_proposal = self.env.resolve_conflict([proposal1, proposal2])
        
        self.assertIsInstance(winning_proposal, ConsensusProposal)
        # Proposal1 should win (more support * higher average confidence)
        self.assertEqual(winning_proposal.proposal_id, "aggressive_strategy")
        
    def test_learning_statistics(self):
        """Test generation of learning statistics."""
        # Add some interaction history
        self.env.interaction_history.append({
            'type': 'observation',
            'observer': 0,
            'model': 1,
            'behavior': 'cooperation',
            'outcome': 0.8,
            'attention_weight': 0.7,
            'timestamp': time.time()
        })
        
        # Add trust connections
        self.env.agents[0].trust_scores[1] = 0.8
        self.env.agents[1].trust_scores[0] = 0.7
        
        stats = self.env.get_learning_statistics()
        
        self.assertEqual(stats['total_agents'], 4)
        self.assertEqual(stats['total_interactions'], 1)
        self.assertIn('cooperation', stats['observation_patterns'])
        self.assertGreater(stats['trust_network_density'], 0.0)
        
    def test_complex_social_learning_scenario(self):
        """Test a complex social learning scenario."""
        # Scenario: Agent 0 demonstrates successful behavior, others observe and learn
        
        # Set up initial trust relationships
        for i in range(1, 4):
            self.env.agents[i].trust_scores[0] = 0.7  # Trust agent 0
            
        # Agent 0 demonstrates successful behavior multiple times
        behaviors = ["efficient_pathfinding", "resource_optimization", "conflict_mediation"]
        outcomes = [0.9, 0.8, 0.85]
        
        for behavior, outcome in zip(behaviors, outcomes):
            for observer_id in range(1, 4):
                self.env.facilitate_observation(
                    observer_id=observer_id,
                    model_id=0,
                    behavior=behavior,
                    outcome=outcome,
                    context={"scenario": "collaborative_task"}
                )
        
        # Check that observers have learned
        total_observations = sum(len(agent.behavioral_memory) for agent in self.env.agents[1:])
        self.assertGreater(total_observations, 0)
        
        # Check that successful patterns were retained
        for agent in self.env.agents[1:]:
            learned_behaviors = list(agent.successful_patterns.keys())
            self.assertTrue(any(behavior in learned_behaviors for behavior in behaviors))
            
        # Verify interaction history
        self.assertEqual(len(self.env.interaction_history), 9)  # 3 behaviors × 3 observers
        
        # Test consensus on learned strategy
        proposal = self.env.agents[0].propose_consensus(
            content={"adopt_strategy": "efficient_pathfinding"},
            confidence=0.9
        )
        
        results = self.env.run_consensus_round(proposal)
        
        # With established trust and successful demonstrations, 
        # consensus should be more likely
        self.assertIn('consensus_reached', results)


if __name__ == '__main__':
    unittest.main()