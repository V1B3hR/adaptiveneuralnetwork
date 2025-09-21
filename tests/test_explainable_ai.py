"""
Test for Explainable Decision Logging System - Phase 3.1

Tests the explainable AI system including decision logging, reasoning chains,
ethical factors, trust calculations, and visualization capabilities.
"""

import json
import os

# Import from the core package
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.explainable_ai import DecisionType, ExplainableDecisionLogger, ReasoningStep


class TestExplainableDecisionLogger(unittest.TestCase):
    def setUp(self):
        """Set up test environment for explainable AI tests."""
        self.logger = ExplainableDecisionLogger(max_logs=100)

    def test_logger_initialization(self):
        """Test explainable decision logger initialization."""
        self.assertEqual(len(self.logger.decision_logs), 0)
        self.assertEqual(len(self.logger.decision_index), 0)
        self.assertEqual(len(self.logger.agent_decisions), 0)
        self.assertEqual(len(self.logger.ethics_violations), 0)

    def test_start_decision_logging(self):
        """Test starting a new decision logging process."""
        agent_id = 1
        decision_type = DecisionType.ETHICAL_ASSESSMENT
        context = {"situation": "resource_allocation", "urgency": "high"}
        inputs = {"available_resources": 100, "requesters": [2, 3, 4]}

        decision_id = self.logger.start_decision_logging(
            agent_id=agent_id, decision_type=decision_type, context=context, inputs=inputs
        )

        # Check that decision was created and indexed
        self.assertIsInstance(decision_id, str)
        self.assertIn(decision_id, self.logger.decision_index)
        self.assertIn(decision_id, self.logger.agent_decisions[agent_id])
        self.assertIn(decision_id, self.logger.decision_type_index[decision_type])

        # Check decision log properties
        decision_log = self.logger.decision_index[decision_id]
        self.assertEqual(decision_log.agent_id, agent_id)
        self.assertEqual(decision_log.decision_type, decision_type)
        self.assertEqual(decision_log.context, context)
        self.assertEqual(decision_log.inputs, inputs)

    def test_add_reasoning_step(self):
        """Test adding reasoning steps to decision chain."""
        # Start decision logging
        decision_id = self.logger.start_decision_logging(
            agent_id=1,
            decision_type=DecisionType.COMMUNICATION,
            context={"recipient": 2},
            inputs={"message": "collaboration_request"},
        )

        # Add reasoning step
        step_id = self.logger.add_reasoning_step(
            decision_id=decision_id,
            step_type=ReasoningStep.ANALYSIS,
            description="Analyzing recipient trust level and communication history",
            inputs={"trust_score": 0.8, "history_length": 15},
            processing={"analysis_method": "weighted_average", "weights": [0.6, 0.4]},
            outputs={"recommendation": "proceed", "confidence": 0.85},
            confidence=0.85,
        )

        # Verify reasoning step was added
        decision_log = self.logger.decision_index[decision_id]
        self.assertEqual(len(decision_log.reasoning_chain), 1)

        reasoning_trace = decision_log.reasoning_chain[0]
        self.assertEqual(reasoning_trace.step_id, step_id)
        self.assertEqual(reasoning_trace.step_type, ReasoningStep.ANALYSIS)
        self.assertEqual(reasoning_trace.confidence, 0.85)
        self.assertIn("trust_score", reasoning_trace.inputs)

    def test_add_ethical_factor(self):
        """Test adding ethical factors to decision."""
        # Start decision logging
        decision_id = self.logger.start_decision_logging(
            agent_id=1,
            decision_type=DecisionType.RESOURCE_ALLOCATION,
            context={"scenario": "emergency_response"},
            inputs={"resources": {"medical": 10, "food": 50}},
        )

        # Add ethical factor
        self.logger.add_ethical_factor(
            decision_id=decision_id,
            factor_name="Preserve Life",
            law_reference="Universal Law #1: Cause No Harm",
            weight=0.9,
            assessment="High priority - emergency medical needs identified",
            compliance_score=0.95,
            rationale="Prioritizing medical resources saves lives in emergency",
        )

        # Add second ethical factor
        self.logger.add_ethical_factor(
            decision_id=decision_id,
            factor_name="Fair Distribution",
            law_reference="Universal Law #3: Pursue Justice",
            weight=0.7,
            assessment="Moderate concern - some inequality in distribution",
            compliance_score=0.75,
            rationale="Distribution favors medical needs but maintains some fairness",
        )

        # Verify ethical factors were added and overall score calculated
        decision_log = self.logger.decision_index[decision_id]
        self.assertEqual(len(decision_log.ethical_factors), 2)

        # Check overall ethics score calculation
        expected_score = (0.95 * 0.9 + 0.75 * 0.7) / (0.9 + 0.7)
        self.assertAlmostEqual(decision_log.overall_ethics_score, expected_score, places=3)

    def test_add_trust_calculation(self):
        """Test adding trust calculation details."""
        # Start decision logging
        decision_id = self.logger.start_decision_logging(
            agent_id=1,
            decision_type=DecisionType.TRUST_EVALUATION,
            context={"target_agent": 3},
            inputs={"recent_interactions": 5},
        )

        # Add trust calculation
        self.logger.add_trust_calculation(
            decision_id=decision_id,
            agent_id=3,
            current_trust=0.75,
            previous_trust=0.70,
            factors={"success_rate": 0.85, "honesty": 0.90, "reliability": 0.80},
            interaction_history_size=25,
            calculation_method="weighted_moving_average",
        )

        # Verify trust calculation was added
        decision_log = self.logger.decision_index[decision_id]
        self.assertEqual(len(decision_log.trust_calculations), 1)

        trust_calc = decision_log.trust_calculations[0]
        self.assertEqual(trust_calc.agent_id, 3)
        self.assertEqual(trust_calc.current_trust, 0.75)
        self.assertAlmostEqual(trust_calc.trust_change, 0.05, places=5)
        self.assertIn("success_rate", trust_calc.factors)

        # Check trust evolution tracking
        self.assertIn(3, self.logger.trust_evolution)
        self.assertEqual(len(self.logger.trust_evolution[3]), 1)

    def test_finalize_decision(self):
        """Test finalizing a decision."""
        # Start decision logging and add some content
        decision_id = self.logger.start_decision_logging(
            agent_id=1,
            decision_type=DecisionType.CONSENSUS_BUILDING,
            context={"proposal": "exploration_strategy"},
            inputs={"participants": [2, 3, 4]},
        )

        # Add reasoning step
        self.logger.add_reasoning_step(
            decision_id,
            ReasoningStep.EVALUATION,
            "Evaluating consensus options",
            {"options": ["A", "B", "C"]},
            {"method": "voting"},
            {"winner": "B"},
            0.8,
        )

        # Add ethical factor with high compliance
        self.logger.add_ethical_factor(
            decision_id,
            "Respect Autonomy",
            "Universal Law #7",
            0.8,
            "Good",
            0.85,
            "All agents consulted",
        )

        # Finalize decision
        decision = {"chosen_strategy": "B", "support_level": 0.75}
        alternatives = [{"strategy": "A"}, {"strategy": "C"}]

        self.logger.finalize_decision(
            decision_id=decision_id,
            decision=decision,
            confidence=0.82,
            alternatives_considered=alternatives,
            dependencies=[],
        )

        # Verify finalization
        decision_log = self.logger.decision_index[decision_id]
        self.assertEqual(decision_log.decision, decision)
        self.assertEqual(decision_log.confidence, 0.82)
        self.assertEqual(len(decision_log.alternatives_considered), 2)
        self.assertGreater(decision_log.processing_duration, 0)

        # Check that it was added to main logs
        self.assertEqual(len(self.logger.decision_logs), 1)
        self.assertEqual(self.logger.decision_statistics[DecisionType.CONSENSUS_BUILDING], 1)

    def test_get_decision_explanation(self):
        """Test generating comprehensive decision explanation."""
        # Create and finalize a complete decision
        decision_id = self._create_complete_decision()

        # Get explanation
        explanation = self.logger.get_decision_explanation(decision_id)

        # Verify explanation structure
        self.assertIn("decision_summary", explanation)
        self.assertIn("reasoning_chain", explanation)
        self.assertIn("ethical_analysis", explanation)
        self.assertIn("trust_considerations", explanation)
        self.assertIn("alternatives", explanation)
        self.assertIn("metadata", explanation)

        # Check decision summary
        summary = explanation["decision_summary"]
        self.assertEqual(summary["id"], decision_id)
        self.assertIn("type", summary)
        self.assertIn("confidence", summary)

        # Check reasoning chain
        self.assertGreater(len(explanation["reasoning_chain"]), 0)
        first_step = explanation["reasoning_chain"][0]
        self.assertIn("step", first_step)
        self.assertIn("type", first_step)
        self.assertIn("description", first_step)

        # Check ethical analysis
        ethics = explanation["ethical_analysis"]
        self.assertIn("overall_score", ethics)
        self.assertIn("compliant", ethics)
        self.assertIn("factors", ethics)

    def test_generate_audit_trail(self):
        """Test generating audit trail."""
        # Create multiple decisions
        decision_ids = []
        for i in range(3):
            decision_id = self._create_complete_decision(agent_id=i + 1)
            decision_ids.append(decision_id)

        # Generate full audit trail
        audit_trail = self.logger.generate_audit_trail()

        self.assertEqual(len(audit_trail), 3)

        # Check audit entry structure
        entry = audit_trail[0]
        self.assertIn("timestamp", entry)
        self.assertIn("decision_id", entry)
        self.assertIn("agent_id", entry)
        self.assertIn("decision_type", entry)
        self.assertIn("ethics_compliant", entry)

        # Test filtered audit trail (by agent)
        agent_1_trail = self.logger.generate_audit_trail(agent_id=1)
        self.assertEqual(len(agent_1_trail), 1)
        self.assertEqual(agent_1_trail[0]["agent_id"], 1)

        # Test filtered audit trail (by decision type)
        ethics_trail = self.logger.generate_audit_trail(
            decision_type=DecisionType.ETHICAL_ASSESSMENT
        )
        self.assertGreaterEqual(len(ethics_trail), 0)

    def test_get_decision_flow_graph(self):
        """Test generating decision flow graph."""
        decision_id = self._create_complete_decision()

        # Get flow graph
        graph = self.logger.get_decision_flow_graph(decision_id)

        # Verify graph structure
        self.assertIn("decision_id", graph)
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        self.assertIn("metadata", graph)

        self.assertEqual(graph["decision_id"], decision_id)
        self.assertGreater(len(graph["nodes"]), 0)
        self.assertGreaterEqual(len(graph["edges"]), 0)

        # Check node structure
        if graph["nodes"]:
            node = graph["nodes"][0]
            self.assertIn("id", node)
            self.assertIn("label", node)
            self.assertIn("type", node)
            self.assertIn("details", node)

    def test_get_analytics_dashboard(self):
        """Test generating analytics dashboard."""
        # Create some decisions
        for i in range(5):
            self._create_complete_decision(agent_id=i % 2 + 1)

        # Get dashboard
        dashboard = self.logger.get_analytics_dashboard()

        # Verify dashboard structure
        self.assertIn("summary", dashboard)
        self.assertIn("distributions", dashboard)
        self.assertIn("recent_violations", dashboard)
        self.assertIn("trust_evolution", dashboard)

        # Check summary
        summary = dashboard["summary"]
        self.assertEqual(summary["total_decisions"], 5)
        self.assertIn("ethics_compliance_rate", summary)
        self.assertIn("average_confidence", summary)

        # Check distributions
        distributions = dashboard["distributions"]
        self.assertIn("decision_types", distributions)
        self.assertIn("agent_activity", distributions)

    def test_export_decision_logs(self):
        """Test exporting decision logs."""
        # Create a few decisions
        for i in range(2):
            self._create_complete_decision(agent_id=i + 1)

        # Export as JSON
        export_json = self.logger.export_decision_logs("json")

        # Should be valid JSON
        data = json.loads(export_json)

        self.assertIn("export_timestamp", data)
        self.assertIn("total_decisions", data)
        self.assertIn("decisions", data)

        self.assertEqual(data["total_decisions"], 2)
        self.assertEqual(len(data["decisions"]), 2)

        # Check decision structure in export
        decision = data["decisions"][0]
        self.assertIn("decision_id", decision)
        self.assertIn("decision_type", decision)
        self.assertIn("reasoning_chain", decision)
        self.assertIn("ethical_factors", decision)

    def test_ethics_violation_tracking(self):
        """Test tracking of ethics violations."""
        # Create decision with low ethics score
        decision_id = self.logger.start_decision_logging(
            agent_id=1,
            decision_type=DecisionType.ETHICAL_ASSESSMENT,
            context={"test": "violation"},
            inputs={},
        )

        # Add ethical factor with low compliance score
        self.logger.add_ethical_factor(
            decision_id, "Test Factor", "Test Law", 1.0, "Poor", 0.3, "Violation test"
        )

        # Finalize decision (should trigger violation detection)
        self.logger.finalize_decision(decision_id, {"result": "violation"}, 0.5)

        # Check that violation was recorded
        self.assertIn(decision_id, self.logger.ethics_violations)

    def test_invalid_decision_id_handling(self):
        """Test handling of invalid decision IDs."""
        invalid_id = "nonexistent_decision"

        # Should raise ValueError for invalid decision ID
        with self.assertRaises(ValueError):
            self.logger.add_reasoning_step(
                invalid_id, ReasoningStep.ANALYSIS, "test", {}, {}, {}, 0.5
            )

        with self.assertRaises(ValueError):
            self.logger.add_ethical_factor(invalid_id, "test", "test", 1.0, "test", 0.5, "test")

        with self.assertRaises(ValueError):
            self.logger.get_decision_explanation(invalid_id)

    def _create_complete_decision(self, agent_id: int = 1) -> str:
        """Helper method to create a complete decision for testing."""
        # Start decision
        decision_id = self.logger.start_decision_logging(
            agent_id=agent_id,
            decision_type=DecisionType.ETHICAL_ASSESSMENT,
            context={"test_scenario": "complete_decision"},
            inputs={"test_input": "value"},
        )

        # Add reasoning steps
        self.logger.add_reasoning_step(
            decision_id,
            ReasoningStep.OBSERVATION,
            "Observing situation",
            {"observations": ["A", "B"]},
            {"method": "direct"},
            {"summary": "observed"},
            0.9,
        )

        self.logger.add_reasoning_step(
            decision_id,
            ReasoningStep.ANALYSIS,
            "Analyzing options",
            {"options": ["X", "Y"]},
            {"method": "comparison"},
            {"best": "X"},
            0.8,
        )

        # Add ethical factor
        self.logger.add_ethical_factor(
            decision_id, "Test Ethics", "Test Law #1", 0.8, "Good", 0.8, "Test rationale"
        )

        # Add trust calculation
        self.logger.add_trust_calculation(
            decision_id, agent_id + 10, 0.7, 0.6, {"factor1": 0.8}, 10, "test_method"
        )

        # Finalize
        self.logger.finalize_decision(
            decision_id, {"result": "test_decision"}, 0.85, [{"alt": "alternative"}]
        )

        return decision_id


if __name__ == "__main__":
    unittest.main()
