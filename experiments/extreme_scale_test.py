import random
from config.network_config import load_network_config
from core.alive_node import AliveLoopNode
from core.capacitor import CapacitorInSpace
from core.network import TunedAdaptiveFieldNetwork
from api_integrations.human_api import fetch_human_signal
from api_integrations.ai_api import fetch_ai_signal
from api_integrations.world_api import fetch_world_signal

def get_real_streams(api_endpoints, node_count):
    types = ["human", "AI", "world"]
    # Only fetch real signals for first 30 nodes to avoid API flooding
    streams = {}
    for node_id in range(min(node_count, 30)):
        signal_type = types[node_id % 3]
        if signal_type == "human":
            streams[node_id] = fetch_human_signal(api_endpoints["human"])
        elif signal_type == "AI":
            streams[node_id] = fetch_ai_signal(api_endpoints["ai"])
        elif signal_type == "world":
            streams[node_id] = fetch_world_signal(api_endpoints["world"])
    return streams

def get_synthetic_streams(node_count):
    types = ["human", "AI", "world"]
    # All nodes receive synthetic signals
    return {
        node_id: (random.choice(types), random.uniform(1, 8))
        for node_id in range(node_count)
    }

def run_extreme_scale_test(steps=10000, node_count=9000, capacitor_count=600, snapshot_interval=500):
    cfg = load_network_config("config/network_config.yaml")
    print(f"Initializing {node_count} nodes...")
    nodes = [
        AliveLoopNode(
            position=[random.uniform(-10, 10), random.uniform(-10, 10)],
            velocity=[random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15)],
            initial_energy=random.uniform(5, 15),
            field_strength=random.uniform(0.8, 1.3),
            node_id=i
        )
        for i in range(node_count)
    ]
    print(f"Initializing {capacitor_count} capacitors...")
    capacitors = [
        CapacitorInSpace(
            position=[random.uniform(-12, 12), random.uniform(-12, 12)],
            capacity=random.uniform(3, 12),
            initial_energy=random.uniform(0, 6)
        )
        for _ in range(capacitor_count)
    ]
    api_endpoints = cfg["api_endpoints"]
    network = TunedAdaptiveFieldNetwork(
        nodes, capacitors,
        api_endpoints={
            i: {"type": ["human", "AI", "world"][i % 3], "url": api_endpoints[["human", "ai", "world"][i % 3]]}
            for i in range(min(node_count, 30))  # Only first 30 nodes get real signals
        }
    )

    results = []
    print("Starting extreme scale test...")

    for step in range(steps):
        # Use real external signals for the first 30 nodes every 50 steps, synthetic for all others
        if step % 100 < 50:
            external_streams = get_real_streams(api_endpoints, node_count)
        else:
            external_streams = get_synthetic_streams(node_count)

        network.step(external_streams)
        # Only save snapshots at intervals for performance
        if step % snapshot_interval == 0 or step == steps - 1:
            print(f"--- Step {step} ---")
            # Save aggregate metrics only
            avg_energy = sum(node.energy for node in nodes) / node_count
            avg_reward = sum(node.reward for node in nodes) / node_count
            avg_anxiety = sum(node.anxiety for node in nodes) / node_count
            avg_cap_energy = sum(cap.energy for cap in capacitors) / capacitor_count
            snapshot = {
                "time": network.time,
                "avg_node_energy": avg_energy,
                "avg_node_reward": avg_reward,
                "avg_node_anxiety": avg_anxiety,
                "avg_capacitor_energy": avg_cap_energy,
            }
            results.append(snapshot)
            print(snapshot)

    print("Extreme scale test complete. Results saved to 'extreme_scale_test_results.json'.")
    import json
    with open("extreme_scale_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_extreme_scale_test(steps=10000, node_count=9000, capacitor_count=600, snapshot_interval=500)
