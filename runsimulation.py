from config.network_config import load_network_config
from core.alive_node import AliveLoopNode
from core.capacitor import CapacitorInSpace
from core.network import TunedAdaptiveFieldNetwork

def main():
    cfg = load_network_config("config/network_config.yaml")
    nodes = [
        AliveLoopNode(
            position=[0, 0],
            velocity=[0.15, 0],
            initial_energy=10,
            field_strength=1.0,
            node_id=0
        ),
        AliveLoopNode(
            position=[1, 2],
            velocity=[-0.08, 0.03],
            initial_energy=5,
            field_strength=1.2,
            node_id=1
        ),
        AliveLoopNode(
            position=[-1, -1],
            velocity=[0.05, 0.09],
            initial_energy=7,
            field_strength=0.9,
            node_id=2
        )
    ]
    capacitors = [
        CapacitorInSpace(position=[0.5, 0.5], capacity=4),
        CapacitorInSpace(position=[-0.5, -0.5], capacity=6),
        CapacitorInSpace(position=[2, 2], capacity=5)
    ]
    # Map API endpoints to node IDs
    api_endpoints = {
        0: {"type": "human", "url": cfg["api_endpoints"]["human"]},
        1: {"type": "AI", "url": cfg["api_endpoints"]["ai"]},
        2: {"type": "world", "url": cfg["api_endpoints"]["world"]}
    }
    network = TunedAdaptiveFieldNetwork(nodes, capacitors, api_endpoints=api_endpoints)

    print("Initial State:")
    network.print_states()
    print("\nSimulating 15 steps with live API streams every 5 steps...\n")
    for step in range(15):
        network.step()
        network.print_states()
    print("\nTest complete.")

if __name__ == "__main__":
    main()
