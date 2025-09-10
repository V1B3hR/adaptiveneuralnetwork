#!/usr/bin/env python3
"""
Demonstration script for the improved CapacitorInSpace class

This script showcases all the improvements and features of the capacitor.py module.
Run this to see the enhanced capabilities in action.
"""

from core.capacitor import CapacitorInSpace
import numpy as np
import logging

def main():
    print("=" * 60)
    print("🔋 CAPACITOR.PY IMPROVEMENTS DEMONSTRATION")
    print("=" * 60)
    
    print("\n🚀 Feature 1: Basic Energy Management")
    print("-" * 40)
    cap = CapacitorInSpace(position=[0, 0], capacity=20.0, initial_energy=10.0)
    print(f"Initial state: {cap}")
    
    absorbed = cap.charge(8.0)
    print(f"Charged 8.0 units, absorbed: {absorbed}")
    print(f"New state: {cap}")
    
    released = cap.discharge(5.0)
    print(f"Discharged 5.0 units, released: {released}")
    print(f"Final state: {cap}")
    
    print("\n🎯 Feature 2: Position Validation & Bounds")
    print("-" * 40)
    bounds = ((-10.0, 10.0), (-15.0, 15.0))
    bounded_cap = CapacitorInSpace(
        position=[5.0, -8.0],
        capacity=15.0,
        bounds=bounds,
        expected_dims=2
    )
    print(f"Bounded capacitor: {bounded_cap}")
    print(f"Bounds: {bounds}")
    
    # Valid position update
    bounded_cap.update_position([8.0, 12.0])
    print(f"Updated position: {bounded_cap.position}")
    
    # Invalid position update (will raise error)
    try:
        bounded_cap.update_position([15.0, 0.0])  # Outside bounds
    except ValueError as e:
        print(f"Bounds violation prevented: {e}")
    
    print("\n🔒 Feature 3: Fixed Position Mode")
    print("-" * 40)
    fixed_cap = CapacitorInSpace(
        position=[2.0, 3.0, 4.0],
        capacity=30.0,
        fixed_position=True
    )
    print(f"Fixed position capacitor: {fixed_cap}")
    
    try:
        fixed_cap.update_position([1.0, 1.0, 1.0])
    except RuntimeError as e:
        print(f"Position update prevented: {e}")
    
    print("\n🧵 Feature 4: Thread Safety")
    print("-" * 40)
    threadsafe_cap = CapacitorInSpace(
        position=[0, 0],
        capacity=100.0,
        initial_energy=50.0,
        thread_safe=True
    )
    print(f"Thread-safe capacitor: {threadsafe_cap}")
    print("✅ Operations are protected by RLock for multi-threading")
    
    print("\n📊 Feature 5: Advanced Monitoring & Serialization")
    print("-" * 40)
    monitor_cap = CapacitorInSpace(
        position=[1.0, 2.0],
        capacity=40.0,
        initial_energy=28.0,
        expected_dims=2,
        bounds=((-5, 5), (-5, 5))
    )
    
    data = monitor_cap.to_dict()
    print("📈 Capacitor telemetry:")
    for key, value in data.items():
        if key == 'soc':
            print(f"  {key}: {value:.1%}")
        else:
            print(f"  {key}: {value}")
    
    print("\n🔧 Feature 6: Configurable Logging")
    print("-" * 40)
    # Create logger with custom verbosity
    logger = logging.getLogger("demo_capacitor")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    debug_cap = CapacitorInSpace(
        position=[0, 0],
        capacity=10.0,
        initial_energy=5.0,
        logger=logger,
        verbosity=logging.DEBUG
    )
    
    print("🔍 Debug logging enabled - operations will show detailed info:")
    debug_cap.charge(2.0)
    debug_cap.discharge(1.0)
    debug_cap.update_position([1.0, 1.0])
    
    print("\n⚡ Feature 7: Energy Safety & Clamping")
    print("-" * 40)
    # Test energy clamping
    safety_cap = CapacitorInSpace(
        position=[0, 0],
        capacity=10.0,
        initial_energy=15.0  # Above capacity
    )
    print(f"Created with energy=15.0, capacity=10.0")
    print(f"Actual energy (clamped): {safety_cap.energy}")
    
    # Test negative capacity
    neg_cap = CapacitorInSpace(
        position=[0, 0],
        capacity=-5.0,  # Invalid
        initial_energy=2.0
    )
    print(f"Created with capacity=-5.0, actual capacity: {neg_cap.capacity}")
    
    print("\n" + "=" * 60)
    print("✅ ALL CAPACITOR.PY IMPROVEMENTS DEMONSTRATED!")
    print("🎉 The enhanced CapacitorInSpace class provides:")
    print("   • Robust energy management with safety checks")
    print("   • Advanced position validation and bounds enforcement")
    print("   • Thread safety for concurrent operations")
    print("   • Configurable logging and monitoring")
    print("   • Flexible constraints and validation options")
    print("   • Full backward compatibility with existing code")
    print("=" * 60)

if __name__ == "__main__":
    main()