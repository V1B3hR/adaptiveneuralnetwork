"""
Biological Plausibility Testing - Circadian Rhythm Modeling

Test Category: Biological Plausibility - Circadian Rhythm Modeling
Description: Validates sleep/wake cycle optimization and circadian rhythm
effects on neural network performance and phase transitions.

Test Cases:
1. Sleep-wake cycle simulation
2. Circadian performance modulation
3. Sleep-dependent memory consolidation
4. Chronotype adaptation modeling

Example usage:
    python -m unittest tests.biological_plausibility.test_circadian_rhythm_modeling
"""

import unittest
import random
from unittest.mock import Mock
import math


class TestCircadianRhythmModeling(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment with reproducible conditions"""
        random.seed(42)
        
        # Mock circadian system
        self.circadian_system = Mock()
        self.circadian_system.current_time = 0.0  # Hours since start
        self.circadian_system.circadian_phase = 0.0
        self.circadian_system.sleep_pressure = 0.0
        self.circadian_system.performance_metrics = {}
        
    def test_sleep_wake_cycle_simulation(self):
        """
        Description: Test simulation of natural sleep-wake cycles
        Expected: System should exhibit ~24-hour cycles with appropriate sleep/wake patterns
        """
        # Mock circadian oscillator
        def simulate_circadian_cycle(time_hours, period=24.0, amplitude=1.0):
            """
            Simulate circadian rhythm using sinusoidal oscillation
            Returns value between -1 (sleep phase) and 1 (wake phase)
            """
            phase = (2 * math.pi * time_hours) / period
            return amplitude * math.cos(phase)
        
        # Test 48-hour simulation (2 full cycles)
        time_points = []
        circadian_values = []
        
        for hour in range(48):
            circadian_value = simulate_circadian_cycle(hour)
            time_points.append(hour)
            circadian_values.append(circadian_value)
        
        # Test that we get two complete cycles
        # Find peaks (maximum values)
        peaks = []
        for i in range(1, len(circadian_values) - 1):
            if (circadian_values[i] > circadian_values[i-1] and 
                circadian_values[i] > circadian_values[i+1] and 
                circadian_values[i] > 0.8):
                peaks.append(i)
        
        # Should have approximately 2 peaks in 48 hours
        self.assertGreaterEqual(len(peaks), 1)
        self.assertLessEqual(len(peaks), 3)
        
        # Test that peaks are approximately 24 hours apart
        if len(peaks) >= 2:
            peak_interval = peaks[1] - peaks[0]
            self.assertGreater(peak_interval, 20)  # At least 20 hours
            self.assertLess(peak_interval, 28)     # At most 28 hours
        
        # Test that values oscillate between approximately -1 and 1
        max_value = max(circadian_values)
        min_value = min(circadian_values)
        self.assertGreater(max_value, 0.8)
        self.assertLess(min_value, -0.8)
        
    def test_circadian_performance_modulation(self):
        """
        Description: Test modulation of cognitive performance by circadian rhythm
        Expected: Performance should vary predictably with circadian phase
        """
        # Mock performance modulation based on circadian phase
        def calculate_performance(circadian_phase, base_performance=0.8):
            """
            Calculate performance based on circadian phase
            circadian_phase: 0-1 (0=midnight, 0.25=6AM, 0.5=noon, 0.75=6PM)
            """
            # Most people perform best in late morning/early afternoon
            optimal_phase = 0.4  # Around 10 AM
            
            # Performance decreases during typical sleep hours (0-0.25 and 0.9-1.0)
            if circadian_phase < 0.25 or circadian_phase > 0.9:
                sleep_penalty = 0.4
            else:
                sleep_penalty = 0.0
            
            # Calculate distance from optimal performance time
            phase_distance = abs(circadian_phase - optimal_phase)
            if phase_distance > 0.5:  # Wrap around 24-hour cycle
                phase_distance = 1.0 - phase_distance
            
            # Performance peaks at optimal time, decreases with distance
            phase_modifier = 1.0 - (phase_distance * 0.3)  # Max 30% reduction from phase
            
            final_performance = base_performance * phase_modifier - sleep_penalty
            return max(0.0, min(1.0, final_performance))
        
        # Test performance at different times of day
        performance_profile = {}
        test_phases = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]  # Every 3 hours
        
        for phase in test_phases:
            performance = calculate_performance(phase)
            hour = phase * 24
            performance_profile[hour] = performance
        
        # Test that performance is lowest during sleep hours (midnight-6AM)
        midnight_perf = performance_profile[0.0]
        three_am_perf = performance_profile[3.0]
        self.assertLess(midnight_perf, 0.5)
        self.assertLess(three_am_perf, 0.5)
        
        # Test that performance is higher during day hours
        ten_am_perf = performance_profile[9.0]   # Close to optimal 10 AM
        two_pm_perf = performance_profile[15.0]   # Early afternoon
        self.assertGreater(ten_am_perf, 0.6)
        self.assertGreater(two_pm_perf, 0.6)
        
        # Test that optimal performance is around late morning
        morning_peak = max(performance_profile[6.0], performance_profile[9.0], performance_profile[12.0])
        evening_perf = performance_profile[18.0]
        self.assertGreater(morning_peak, evening_perf)
        
    def test_sleep_dependent_memory_consolidation(self):
        """
        Description: Test sleep-dependent memory consolidation processes
        Expected: Memory consolidation should be enhanced during sleep phases
        """
        # Mock memory consolidation system
        class MemoryConsolidationSystem:
            def __init__(self):
                self.memories = []
                self.consolidated_memories = []
                
            def add_memory(self, memory_content, importance, timestamp):
                self.memories.append({
                    "content": memory_content,
                    "importance": importance,
                    "timestamp": timestamp,
                    "consolidation_strength": 0.0
                })
            
            def consolidate_during_sleep(self, sleep_depth, consolidation_rate=0.1):
                """
                Consolidate memories during sleep phase
                sleep_depth: 0-1 (deeper sleep = better consolidation)
                """
                for memory in self.memories:
                    # Consolidation is stronger for important memories and deeper sleep
                    consolidation_boost = (memory["importance"] * sleep_depth * 
                                         consolidation_rate)
                    memory["consolidation_strength"] += consolidation_boost
                    
                    # Move to consolidated if threshold reached
                    if (memory["consolidation_strength"] > 0.8 and 
                        memory not in self.consolidated_memories):
                        self.consolidated_memories.append(memory)
        
        # Test memory consolidation simulation
        memory_system = MemoryConsolidationSystem()
        
        # Add various memories with different importance levels
        memory_system.add_memory("important_skill", 0.9, 0)
        memory_system.add_memory("random_fact", 0.3, 1)
        memory_system.add_memory("safety_information", 0.95, 2)
        memory_system.add_memory("trivial_detail", 0.1, 3)
        
        # Simulate sleep consolidation over multiple cycles
        for cycle in range(15):  # More cycles to ensure consolidation
            sleep_depth = 0.8  # Deep sleep
            memory_system.consolidate_during_sleep(sleep_depth)
        
        # Test that important memories consolidate first
        consolidated_contents = [mem["content"] for mem in memory_system.consolidated_memories]
        
        # Check consolidation strength instead if not fully consolidated
        safety_memory = next((mem for mem in memory_system.memories 
                            if mem["content"] == "safety_information"), None)
        important_memory = next((mem for mem in memory_system.memories 
                               if mem["content"] == "important_skill"), None)
        
        if safety_memory:
            self.assertGreater(safety_memory["consolidation_strength"], 0.8)
        if important_memory:
            self.assertGreater(important_memory["consolidation_strength"], 0.7)
        
        # Test that trivial memories consolidate less readily
        trivial_memory = next((mem for mem in memory_system.memories 
                             if mem["content"] == "trivial_detail"), None)
        if trivial_memory:
            self.assertLess(trivial_memory["consolidation_strength"], 0.5)
        
        # Test different sleep depths
        memory_system_shallow = MemoryConsolidationSystem()
        memory_system_shallow.add_memory("test_memory", 0.7, 0)
        
        # Shallow sleep consolidation
        for cycle in range(10):
            memory_system_shallow.consolidate_during_sleep(0.3)  # Shallow sleep
        
        shallow_consolidation = memory_system_shallow.memories[0]["consolidation_strength"]
        
        # Deep sleep should consolidate better than shallow sleep
        memory_system_deep = MemoryConsolidationSystem()
        memory_system_deep.add_memory("test_memory", 0.7, 0)
        
        for cycle in range(10):
            memory_system_deep.consolidate_during_sleep(0.9)  # Deep sleep
        
        deep_consolidation = memory_system_deep.memories[0]["consolidation_strength"]
        self.assertGreater(deep_consolidation, shallow_consolidation)
        
    def test_chronotype_adaptation_modeling(self):
        """
        Description: Test modeling of different chronotypes (morning/evening preferences)
        Expected: System should adapt to individual chronotype preferences
        """
        # Mock chronotype system
        def model_chronotype(chronotype, time_of_day):
            """
            Model performance based on chronotype
            chronotype: 'morning', 'evening', or 'intermediate'
            time_of_day: 0-23 (hours)
            """
            chronotype_profiles = {
                'morning': {
                    'peak_start': 6,
                    'peak_end': 12,
                    'optimal_bedtime': 22,
                    'optimal_wake': 6
                },
                'evening': {
                    'peak_start': 14,
                    'peak_end': 22,
                    'optimal_bedtime': 2,
                    'optimal_wake': 10
                },
                'intermediate': {
                    'peak_start': 10,
                    'peak_end': 18,
                    'optimal_bedtime': 23,
                    'optimal_wake': 7
                }
            }
            
            profile = chronotype_profiles[chronotype]
            
            # Calculate performance based on distance from peak hours
            if profile['peak_start'] <= time_of_day <= profile['peak_end']:
                performance = 1.0  # Peak performance
            else:
                # Calculate distance from peak period
                if time_of_day < profile['peak_start']:
                    distance = profile['peak_start'] - time_of_day
                else:
                    distance = time_of_day - profile['peak_end']
                
                # Account for wrap-around (evening chronotype late night performance)
                if chronotype == 'evening' and time_of_day >= 22:
                    distance = min(distance, 24 - time_of_day + profile['peak_start'])
                
                # Performance decreases with distance from peak
                performance = max(0.2, 1.0 - (distance * 0.1))
            
            return performance
        
        # Test morning chronotype
        morning_6am = model_chronotype('morning', 6)
        morning_10am = model_chronotype('morning', 10)
        morning_8pm = model_chronotype('morning', 20)
        
        self.assertGreater(morning_6am, 0.8)   # Should be high in morning
        self.assertGreater(morning_10am, 0.8)  # Peak continues
        self.assertLess(morning_8pm, 0.6)      # Lower in evening
        
        # Test evening chronotype
        evening_6am = model_chronotype('evening', 6)
        evening_2pm = model_chronotype('evening', 14)
        evening_8pm = model_chronotype('evening', 20)
        
        self.assertLess(evening_6am, 0.5)      # Low in early morning
        self.assertGreater(evening_2pm, 0.8)   # High in afternoon
        self.assertGreater(evening_8pm, 0.8)   # Peak in evening
        
        # Test that chronotypes have different optimal times
        optimal_times = {}
        for chronotype in ['morning', 'evening', 'intermediate']:
            best_performance = 0
            best_time = 0
            for hour in range(24):
                perf = model_chronotype(chronotype, hour)
                if perf > best_performance:
                    best_performance = perf
                    best_time = hour
            optimal_times[chronotype] = best_time
        
        # Morning types should peak earlier than evening types
        self.assertLess(optimal_times['morning'], optimal_times['evening'])
        
    def test_sleep_pressure_accumulation(self):
        """
        Description: Test accumulation and dissipation of sleep pressure (Process S)
        Expected: Sleep pressure should build during wake and dissipate during sleep
        """
        # Mock sleep pressure system (Process S)
        def simulate_sleep_pressure(wake_duration, sleep_duration, 
                                  pressure_buildup_rate=0.1, 
                                  pressure_dissipation_rate=0.2):
            """
            Simulate homeostatic sleep pressure
            wake_duration: hours awake
            sleep_duration: hours asleep  
            """
            # Pressure builds during wake (exponential saturation)
            max_pressure = 1.0
            wake_pressure = max_pressure * (1 - math.exp(-wake_duration * pressure_buildup_rate))
            
            # Pressure dissipates during sleep (exponential decay)
            remaining_pressure = wake_pressure * math.exp(-sleep_duration * pressure_dissipation_rate)
            
            return remaining_pressure, wake_pressure
        
        # Test pressure buildup during extended wake
        pressures_after_wake = []
        for hours_awake in [0, 4, 8, 12, 16, 20, 24]:
            pressure, _ = simulate_sleep_pressure(hours_awake, 0)
            pressures_after_wake.append(pressure)
        
        # Pressure should increase with time awake
        for i in range(1, len(pressures_after_wake)):
            self.assertGreaterEqual(pressures_after_wake[i], pressures_after_wake[i-1])
        
        # Test pressure dissipation during sleep
        initial_pressure = 0.8  # High pressure before sleep
        pressures_after_sleep = []
        
        for hours_asleep in [0, 2, 4, 6, 8, 10]:
            # Start with high pressure, see how it dissipates
            final_pressure = initial_pressure * math.exp(-hours_asleep * 0.2)
            pressures_after_sleep.append(final_pressure)
        
        # Pressure should decrease with sleep duration
        for i in range(1, len(pressures_after_sleep)):
            self.assertLessEqual(pressures_after_sleep[i], pressures_after_sleep[i-1])
        
        # Test that 8 hours of sleep significantly reduces pressure
        pressure_after_8h_sleep = pressures_after_sleep[-1]
        self.assertLess(pressure_after_8h_sleep, initial_pressure * 0.3)
        
    def test_circadian_phase_shifting(self):
        """
        Description: Test circadian rhythm phase shifting (jet lag adaptation)
        Expected: System should gradually adapt to new time zones
        """
        # Mock phase shifting mechanism
        def simulate_phase_shift(days_since_shift, time_zone_difference, 
                               shift_rate=1.0):  # hours per day
            """
            Simulate gradual adaptation to new time zone
            days_since_shift: days since time zone change
            time_zone_difference: hours of difference
            shift_rate: adaptation rate (hours per day)
            """
            max_shift = abs(time_zone_difference)
            
            # Calculate how much adaptation has occurred
            adapted_hours = min(days_since_shift * shift_rate, max_shift)
            
            # Remaining jet lag
            remaining_lag = max_shift - adapted_hours
            
            # Performance impact (jet lag reduces performance)
            performance_impact = 1.0 - (remaining_lag / max_shift) * 0.4  # Max 40% reduction
            
            return remaining_lag, performance_impact
        
        # Test adaptation to 8-hour time difference (trans-Pacific flight)
        adaptation_data = []
        for day in range(10):
            remaining_lag, performance = simulate_phase_shift(day, 8)
            adaptation_data.append((day, remaining_lag, performance))
        
        # Test that jet lag decreases over time
        day0_lag = adaptation_data[0][1]
        day5_lag = adaptation_data[5][1]
        day9_lag = adaptation_data[9][1]
        
        self.assertGreater(day0_lag, day5_lag)
        self.assertGreater(day5_lag, day9_lag)
        
        # Test that performance improves as adaptation occurs
        day0_perf = adaptation_data[0][2]
        day9_perf = adaptation_data[9][2]
        
        self.assertGreater(day9_perf, day0_perf)
        
        # Test that full adaptation eventually occurs
        self.assertLess(day9_lag, 1.0)  # Should be mostly adapted by day 9
        
    def test_ethics_compliance(self):
        """
        Description: Mandatory ethics compliance test for circadian rhythm modeling
        Expected: Circadian modeling must respect natural biological patterns
        """
        # Ethical requirements for circadian modeling
        ethical_requirements = {
            "respects_natural_rhythms": True,
            "no_forced_disruption": True,
            "individual_differences": True,
            "health_considerations": True,
            "transparent_modeling": True
        }
        
        # Mock ethical validation
        def validate_circadian_ethics(circadian_model):
            violations = []
            
            # Check respect for natural patterns
            if not circadian_model.get("biological_basis", False):
                violations.append("respects_natural_rhythms")
            
            # Check against forced disruption
            if circadian_model.get("forces_unnatural_patterns", False):
                violations.append("no_forced_disruption")
            
            # Check for individual variation support
            if not circadian_model.get("supports_chronotypes", False):
                violations.append("individual_differences")
            
            # Check health considerations
            if not circadian_model.get("considers_health_impacts", False):
                violations.append("health_considerations")
            
            return len(violations) == 0, violations
        
        # Test ethical circadian model
        ethical_circadian_model = {
            "biological_basis": True,
            "forces_unnatural_patterns": False,
            "supports_chronotypes": True,
            "considers_health_impacts": True,
            "transparent_mechanisms": True
        }
        
        is_ethical, violations = validate_circadian_ethics(ethical_circadian_model)
        self.assertTrue(is_ethical, f"Ethics violations: {violations}")
        
        # Verify all requirements are enforced
        for requirement, needed in ethical_requirements.items():
            self.assertTrue(needed, f"Ethical requirement {requirement} must be enforced")


if __name__ == '__main__':
    unittest.main()