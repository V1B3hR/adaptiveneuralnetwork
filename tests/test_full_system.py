import matplotlib.pyplot as plt
import numpy as np
from core.alive_node import AliveLoopNode, Memory


# Simulating a Capacitor class for interactions
class MockCapacitor:
    def __init__(self, position, energy, capacity):
        self.position = np.array(position, dtype=float)
        self.energy = energy
        self.capacity = capacity


def simulate_full_system():
    """Simulate the full system with multiple nodes and capacitors"""

    # Initialize nodes
    nodes = [
        AliveLoopNode(position=(0, 0), velocity=(1, 0), initial_energy=15.0, field_strength=1.0, node_id=1),
        AliveLoopNode(position=(5, 5), velocity=(0, -1), initial_energy=10.0, field_strength=0.8, node_id=2),
    ]

    # Establish trust relationships
    nodes[0].trust_network = {2: 0.8}
    nodes[1].trust_network = {1: 0.6}

    # Create capacitors
    capacitors = [
        MockCapacitor(position=(10, 10), energy=20, capacity=30),
        MockCapacitor(position=(-10, -10), energy=15, capacity=25),
    ]

    # Simulation parameters
    time_steps = 100
    time_interval = 0.5  # Simulates half-hour intervals
    energy_levels = {node.node_id: [] for node in nodes}
    positions = {node.node_id: [] for node in nodes}

    # Run the simulation
    for t in range(time_steps):
        current_time = (t * time_interval) % 24  # Simulate a 24-hour clock

        for node in nodes:
            # Step through phases
            node.step_phase(current_time=current_time)

            # Move the node
            node.move()

            # Interact with capacitors
            for capacitor in capacitors:
                node.interact_with_capacitor(capacitor)

            # Replay memories with other nodes
            node.replay_memories(nodes)

            # Store data for analysis
            energy_levels[node.node_id].append(node.energy)
            positions[node.node_id].append(node.position.copy())

    # Plot results
    plot_simulation_results(nodes, capacitors, energy_levels, positions)


def plot_simulation_results(nodes, capacitors, energy_levels, positions):
    """Plot the results of the simulation"""

    # Plot movement
    plt.figure(figsize=(12, 6))
    for node in nodes:
        positions_array = np.array(positions[node.node_id])
        plt.plot(positions_array[:, 0], positions_array[:, 1], label=f"Node {node.node_id} Path")
    for i, capacitor in enumerate(capacitors):
        plt.scatter(capacitor.position[0], capacitor.position[1], color="red", label=f"Capacitor {i + 1}", marker="x")

    plt.title("Node Movement and Capacitor Locations")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()

    # Plot energy levels
    plt.figure(figsize=(12, 6))
    for node in nodes:
        plt.plot(range(len(energy_levels[node.node_id])), energy_levels[node.node_id], label=f"Node {node.node_id} Energy")
    plt.title("Energy Levels Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    simulate_full_system()
