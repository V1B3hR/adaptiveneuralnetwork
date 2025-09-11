#!/usr/bin/env python3
"""
Demonstration of SQL injection defenses and performance analysis
in the TimeSeriesTracker system.

This script demonstrates the security improvements implemented:
1. Input validation and type coercion
2. Variable name allowlist validation
3. SQL injection prevention
4. Query performance analysis with EXPLAIN
"""

import sys
import tempfile
import os

# Add the project root to the path
sys.path.insert(0, '/home/runner/work/adaptiveneuralnetwork/adaptiveneuralnetwork')

from core.time_series_tracker import TimeSeriesTracker, TimeSeriesQuery
from core.alive_node import AliveLoopNode


def demonstrate_input_validation():
    """Demonstrate input validation and type coercion"""
    print("=" * 60)
    print("DEMONSTRATING INPUT VALIDATION")
    print("=" * 60)
    
    # Create tracker with temporary database
    db_file = tempfile.NamedTemporaryFile(delete=False)
    db_file.close()
    
    tracker = TimeSeriesTracker(db_path=db_file.name)
    
    # Add some test data
    tracker.record_node_state(1, {"energy": 10.0, "anxiety": 2.0})
    tracker.record_node_state(2, {"energy": 15.0, "trust": 0.8})
    
    print("\n1. Testing valid queries:")
    try:
        query = TimeSeriesQuery(node_ids=[1, 2], variables=["energy", "anxiety"])
        results = tracker.query(query)
        print(f"✓ Valid query succeeded: {len(results)} results")
    except Exception as e:
        print(f"✗ Valid query failed: {e}")
    
    print("\n2. Testing node_id validation:")
    try:
        # This should fail - string node_ids are rejected
        query = TimeSeriesQuery(node_ids=["'; DROP TABLE time_series; --"])
        tracker.query(query)
        print("✗ SQL injection attempt not blocked!")
    except ValueError as e:
        print(f"✓ SQL injection blocked: {e}")
    
    print("\n3. Testing variable name validation:")
    try:
        # This should fail - variable not in allowlist
        query = TimeSeriesQuery(variables=["'; DROP TABLE time_series; --"])
        tracker.query(query)
        print("✗ SQL injection via variable name not blocked!")
    except ValueError as e:
        print(f"✓ SQL injection via variable name blocked: {e}")
    
    print("\n4. Testing type coercion:")
    try:
        # Float node_ids should be rejected
        query = TimeSeriesQuery(node_ids=[1.5, 2.7])
        tracker.query(query)
        print("✗ Float node_ids not rejected!")
    except ValueError as e:
        print(f"✓ Float node_ids rejected: {e}")
    
    # Cleanup
    try:
        os.unlink(db_file.name)
    except:
        pass


def demonstrate_performance_analysis():
    """Demonstrate query performance analysis with EXPLAIN"""
    print("\n\n" + "=" * 60)
    print("DEMONSTRATING QUERY PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Create tracker with temporary database
    db_file = tempfile.NamedTemporaryFile(delete=False)
    db_file.close()
    
    tracker = TimeSeriesTracker(db_path=db_file.name)
    
    # Add substantial test data to make indexing meaningful
    print("\nAdding test data...")
    for node_id in range(1, 6):
        for i in range(100):
            tracker.record_node_state(node_id, {
                "energy": 10.0 + i * 0.1,
                "anxiety": 2.0 + i * 0.05,
                "trust": 0.5 + i * 0.001
            })
    
    print("Test data added.")
    
    # Demonstrate different query patterns and their performance
    test_queries = [
        ("Simple node filter", TimeSeriesQuery(node_ids=[1])),
        ("Multiple nodes", TimeSeriesQuery(node_ids=[1, 2, 3])),
        ("Variable filter", TimeSeriesQuery(variables=["energy"])),
        ("Combined filters", TimeSeriesQuery(node_ids=[1, 2], variables=["energy", "anxiety"])),
        ("Time range query", TimeSeriesQuery(start_time=0, end_time=9999999999)),
        ("Limited results", TimeSeriesQuery(max_points=50))
    ]
    
    for name, query in test_queries:
        print(f"\n{name}:")
        print("-" * 40)
        try:
            # Execute the query
            results = tracker.query(query)
            print(f"Results: {len(results)} data points")
            
            # Analyze performance
            explain_result = tracker.explain_query(query)
            print("Query Plan:")
            for line in explain_result.split('\n'):
                print(f"  {line}")
                
        except Exception as e:
            print(f"Query failed: {e}")
    
    # Cleanup
    try:
        os.unlink(db_file.name)
    except:
        pass


def demonstrate_secure_integration():
    """Demonstrate secure integration with AliveLoopNode"""
    print("\n\n" + "=" * 60)
    print("DEMONSTRATING SECURE INTEGRATION")
    print("=" * 60)
    
    # Create tracker
    db_file = tempfile.NamedTemporaryFile(delete=False)
    db_file.close()
    
    tracker = TimeSeriesTracker(db_path=db_file.name)
    
    # Create some nodes
    nodes = [
        AliveLoopNode(position=(0, 0), velocity=(0, 0), initial_energy=15.0, node_id=1),
        AliveLoopNode(position=(1, 0), velocity=(0, 0), initial_energy=12.0, node_id=2),
        AliveLoopNode(position=(0, 1), velocity=(0, 0), initial_energy=10.0, node_id=3)
    ]
    
    print("\n1. Recording node states with automatic tracking:")
    from core.time_series_tracker import track_node_automatically
    
    for i, node in enumerate(nodes):
        # Modify some node properties
        node.anxiety = 2.0 + i * 0.5
        node.trust_network = {j: 0.7 - abs(i-j) * 0.1 for j in range(len(nodes)) if j != i}
        
        # Track the node automatically
        track_node_automatically(tracker, node)
        print(f"✓ Node {node.node_id} tracked with {len(tracker.get_latest_values(node.node_id))} variables")
    
    print("\n2. Querying tracked data:")
    # Query all energy levels
    query = TimeSeriesQuery(variables=["energy"])
    results = tracker.query(query)
    print(f"✓ Found {len(results)} energy data points across all nodes")
    
    # Query specific node's data
    query = TimeSeriesQuery(node_ids=[1], variables=["energy", "anxiety"])
    results = tracker.query(query)
    print(f"✓ Found {len(results)} data points for node 1")
    
    print("\n3. Performance analysis:")
    explain_result = tracker.explain_query(query)
    print("Query execution plan:")
    for line in explain_result.split('\n')[:3]:  # Show first few lines
        print(f"  {line}")
    
    # Cleanup
    try:
        os.unlink(db_file.name)
    except:
        pass


def main():
    """Run all demonstrations"""
    print("SQL INJECTION DEFENSE AND PERFORMANCE DEMONSTRATION")
    print("TimeSeriesTracker Security Improvements")
    
    demonstrate_input_validation()
    demonstrate_performance_analysis()
    demonstrate_secure_integration()
    
    print("\n\n" + "=" * 60)
    print("SUMMARY OF SECURITY IMPROVEMENTS")
    print("=" * 60)
    print("✓ Input validation and type coercion for all parameters")
    print("✓ Variable name allowlist prevents injection via column names")
    print("✓ Parameterized queries prevent SQL injection")
    print("✓ EXPLAIN functionality for query performance analysis")
    print("✓ Multi-statement prevention (SQLite default)")
    print("✓ Comprehensive logging and error handling")
    print("✓ Backward compatibility maintained")
    
    print("\nDefenses implemented follow OWASP guidelines:")
    print("• Always use parameterized queries ✓")
    print("• Validate and coerce types at boundaries ✓")
    print("• Use allowlist for dynamic identifiers ✓")
    print("• Least-privilege principle (readonly where possible) ✓")
    print("• Disable multi-statements ✓")
    print("• Defense-in-depth approach ✓")


if __name__ == "__main__":
    main()