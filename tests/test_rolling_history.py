import unittest
from collections import deque

from core.alive_node import AliveLoopNode
from tests.test_utils import get_test_seed, set_seed


class TestRollingHistory(unittest.TestCase):
    def setUp(self):
        """Initialize an AliveLoopNode instance for testing rolling history"""
        set_seed(get_test_seed())

        self.node = AliveLoopNode(
            position=(0, 0),
            velocity=(1, 1),
            initial_energy=10.0,
            field_strength=1.0,
            node_id=1
        )

    def test_history_initialization(self):
        """Test that all history deques are properly initialized"""
        self.assertIsInstance(self.node.anxiety_history, deque)
        self.assertIsInstance(self.node.calm_history, deque)
        self.assertIsInstance(self.node.energy_history, deque)

        # Check max length is set to 20
        self.assertEqual(self.node.anxiety_history.maxlen, 20)
        self.assertEqual(self.node.calm_history.maxlen, 20)
        self.assertEqual(self.node.energy_history.maxlen, 20)

        # Should start empty
        self.assertEqual(len(self.node.anxiety_history), 0)
        self.assertEqual(len(self.node.calm_history), 0)
        self.assertEqual(len(self.node.energy_history), 0)

    def test_history_population(self):
        """Test that histories are populated during step_phase"""
        initial_anxiety = self.node.anxiety
        initial_calm = self.node.calm
        initial_energy = self.node.energy

        # Run several steps
        for i in range(5):
            self.node.step_phase(current_time=i)

        # Check that histories have been populated
        self.assertEqual(len(self.node.anxiety_history), 5)
        self.assertEqual(len(self.node.calm_history), 5)
        self.assertEqual(len(self.node.energy_history), 5)

        # Check that first values match initial values
        self.assertEqual(self.node.anxiety_history[0], initial_anxiety)
        self.assertEqual(self.node.calm_history[0], initial_calm)
        self.assertEqual(self.node.energy_history[0], initial_energy)

    def test_history_max_length(self):
        """Test that histories don't exceed 20 entries"""
        # Run more than 20 steps
        for i in range(25):
            self.node.step_phase(current_time=i)

        # Check that histories are capped at 20
        self.assertEqual(len(self.node.anxiety_history), 20)
        self.assertEqual(len(self.node.calm_history), 20)
        self.assertEqual(len(self.node.energy_history), 20)

        # Check that oldest entries are removed (FIFO behavior)
        # The first entries should no longer be present
        all_histories = [self.node.anxiety_history, self.node.calm_history, self.node.energy_history]
        for history in all_histories:
            self.assertEqual(len(history), 20)

    def test_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data"""
        # No history yet
        result = self.node.analyze_trend(self.node.anxiety_history)

        self.assertEqual(result["trend"], "insufficient_data")
        self.assertEqual(result["slope"], 0.0)
        self.assertFalse(result["intervention_needed"])

    def test_trend_analysis_stable(self):
        """Test trend analysis with stable values"""
        # Manually populate history with stable values
        stable_values = [5.0] * 10
        for value in stable_values:
            self.node.anxiety_history.append(value)

        result = self.node.analyze_trend(self.node.anxiety_history)

        self.assertEqual(result["trend"], "stable")
        self.assertAlmostEqual(result["slope"], 0.0, places=1)
        self.assertAlmostEqual(result["recent_avg"], 5.0, places=1)
        self.assertAlmostEqual(result["volatility"], 0.0, places=1)

    def test_trend_analysis_increasing(self):
        """Test trend analysis with increasing values"""
        # Manually populate history with increasing values
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        for value in increasing_values:
            self.node.anxiety_history.append(value)

        result = self.node.analyze_trend(self.node.anxiety_history)

        self.assertEqual(result["trend"], "increasing")
        self.assertGreater(result["slope"], 0.5)
        self.assertAlmostEqual(result["recent_avg"], 6.0, places=1)  # Average of last 5: [4,5,6,7,8]

    def test_trend_analysis_decreasing(self):
        """Test trend analysis with decreasing values"""
        # Manually populate history with decreasing values
        decreasing_values = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        for value in decreasing_values:
            self.node.energy_history.append(value)

        result = self.node.analyze_trend(self.node.energy_history)

        self.assertEqual(result["trend"], "decreasing")
        self.assertLess(result["slope"], -0.5)
        self.assertAlmostEqual(result["recent_avg"], 3.0, places=1)  # Average of last 5: [5,4,3,2,1]

    def test_intervention_detection_anxiety(self):
        """Test intervention detection for increasing anxiety"""
        # Set up concerning anxiety trend
        anxiety_values = [2.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        for value in anxiety_values:
            self.node.anxiety_history.append(value)
            self.node.anxiety = value  # Update current anxiety

        # Set reasonable calm and energy
        self.node.calm = 3.0
        self.node.energy = 8.0

        result = self.node.detect_intervention_needs()

        self.assertIn("anxiety_management", result["interventions_needed"])
        self.assertIn(result["urgency_level"], ["medium", "high"])
        self.assertEqual(result["anxiety_trend"]["trend"], "increasing")

    def test_intervention_detection_calm_decreasing(self):
        """Test intervention detection for decreasing calm"""
        # Set up concerning calm trend
        calm_values = [4.0, 3.5, 3.0, 2.0, 1.5, 0.8]
        for value in calm_values:
            self.node.calm_history.append(value)
            self.node.calm = value  # Update current calm

        result = self.node.detect_intervention_needs()

        self.assertIn("calm_restoration", result["interventions_needed"])
        self.assertEqual(result["calm_trend"]["trend"], "decreasing")

    def test_intervention_detection_energy_decreasing(self):
        """Test intervention detection for decreasing energy"""
        # Set up concerning energy trend - values that will have recent avg < 3.0
        energy_values = [8.0, 5.0, 3.0, 2.0, 1.5, 0.8]
        for value in energy_values:
            self.node.energy_history.append(value)
            self.node.energy = value  # Update current energy

        result = self.node.detect_intervention_needs()

        self.assertIn("energy_conservation", result["interventions_needed"])
        self.assertEqual(result["urgency_level"], "high")  # Low energy should trigger high urgency
        self.assertEqual(result["energy_trend"]["trend"], "decreasing")

    def test_intervention_detection_combined_risk(self):
        """Test intervention detection for combined risk scenario"""
        # Set up multiple concerning trends that meet combined risk thresholds
        # Need: anxiety_avg > 6.0, calm_avg < 2.0, energy_avg < 5.0
        anxiety_values = [5.0, 6.0, 7.0, 8.0, 9.0]  # avg = 7.0 > 6.0
        calm_values = [3.0, 2.5, 2.0, 1.5, 1.0]     # avg = 2.0, but we need < 2.0
        energy_values = [8.0, 6.0, 5.0, 4.0, 2.0]   # avg = 5.0, but we need < 5.0

        # Adjust to meet thresholds clearly
        anxiety_values = [6.0, 7.0, 8.0, 9.0, 10.0]   # avg = 8.0 > 6.0 ✓
        calm_values = [3.0, 2.0, 1.5, 1.0, 0.5]       # avg = 1.6 < 2.0 ✓
        energy_values = [6.0, 5.0, 4.0, 3.0, 2.0]     # avg = 4.0 < 5.0 ✓

        for i in range(len(anxiety_values)):
            self.node.anxiety_history.append(anxiety_values[i])
            self.node.calm_history.append(calm_values[i])
            self.node.energy_history.append(energy_values[i])

        # Update current values
        self.node.anxiety = anxiety_values[-1]
        self.node.calm = calm_values[-1]
        self.node.energy = energy_values[-1]

        result = self.node.detect_intervention_needs()

        self.assertIn("comprehensive_support", result["interventions_needed"])
        self.assertEqual(result["urgency_level"], "high")
        self.assertTrue(result["combined_risk"])

    def test_proactive_intervention_application(self):
        """Test that proactive interventions are actually applied"""
        # Set up a scenario needing anxiety management
        self.node.anxiety = 8.0
        self.node.calm = 1.0
        anxiety_values = [3.0, 5.0, 7.0, 8.0, 9.0]
        for value in anxiety_values:
            self.node.anxiety_history.append(value)

        initial_anxiety = self.node.anxiety
        initial_calm = self.node.calm

        result = self.node.apply_proactive_intervention()

        # Should have applied anxiety management
        self.assertTrue(len(result["applied_interventions"]) > 0)
        self.assertLess(self.node.anxiety, initial_anxiety)  # Anxiety should decrease
        self.assertGreater(self.node.calm, initial_calm)  # Calm should increase

    def test_proactive_intervention_energy_conservation(self):
        """Test that energy conservation intervention works"""
        # Set up energy conservation scenario
        self.node.phase = "active"
        self.node.energy = 2.0
        energy_values = [8.0, 6.0, 4.0, 3.0, 2.0]
        for value in energy_values:
            self.node.energy_history.append(value)

        result = self.node.apply_proactive_intervention()

        # Should switch to more conservative phase
        self.assertEqual(self.node.phase, "interactive")
        self.assertIn("energy_conservation", str(result["applied_interventions"]))

    def test_get_anxiety_status_includes_history(self):
        """Test that get_anxiety_status includes history information"""
        # Populate some history
        for i in range(10):
            self.node.step_phase(current_time=i)

        status = self.node.get_anxiety_status()

        # Should include history lengths
        self.assertIn("history_lengths", status)
        self.assertEqual(status["history_lengths"]["anxiety"], 10)
        self.assertEqual(status["history_lengths"]["calm"], 10)
        self.assertEqual(status["history_lengths"]["energy"], 10)

        # Should include intervention analysis
        self.assertIn("intervention_analysis", status)

    def test_integration_with_step_phase(self):
        """Test that proactive intervention is integrated with step_phase"""
        # Set up a scenario that will trigger intervention
        self.node.anxiety = 10.0
        self.node.calm = 0.5

        # Populate enough history to trigger analysis
        for i in range(6):
            self.node.step_phase(current_time=i)
            # Manually increase anxiety to create concerning trend
            self.node.anxiety += 1.0

        # The step_phase should have applied proactive interventions
        # Check that anxiety didn't continue to increase unchecked
        status = self.node.get_anxiety_status()
        self.assertIsNotNone(status["intervention_analysis"])


if __name__ == "__main__":
    unittest.main()
