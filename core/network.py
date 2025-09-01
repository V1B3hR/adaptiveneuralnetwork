from api_integrations.human_api import fetch_human_signal
from api_integrations.ai_api import fetch_ai_signal
from api_integrations.world_api import fetch_world_signal

class TunedAdaptiveFieldNetwork:
    def __init__(self, nodes, capacitors, api_endpoints=None):
        self.nodes = nodes
        self.capacitors = capacitors
        self.time = 0
        self.api_endpoints = api_endpoints or {}

    def fetch_external_streams(self):
        external_streams = {}
        for node in self.nodes:
            node_id = node.node_id
            if node_id in self.api_endpoints:
                endpoint = self.api_endpoints[node_id]
                if endpoint["type"] == "human":
                    external_streams[node_id] = fetch_human_signal(endpoint["url"])
                elif endpoint["type"] == "AI":
                    external_streams[node_id] = fetch_ai_signal(endpoint["url"])
                elif endpoint["type"] == "world":
                    external_streams[node_id] = fetch_world_signal(endpoint["url"])
        return external_streams

    def step(self, external_streams=None):
        self.time += 1
        for node in self.nodes:
            node.step_phase(self.time)
        for node in self.nodes:
            node.seek_capacitors(self.capacitors)
        for node in self.nodes:
            node.move()
        for node in self.nodes:
            node.update_size()
        for node in self.nodes:
            for capacitor in self.capacitors:
                node.interact_with_capacitor(capacitor)
        # Fetch and deliver external signals
        if external_streams is None and self.time % 5 == 0:
            external_streams = self.fetch_external_streams()
        if external_streams:
            for node in self.nodes:
                if node.node_id in external_streams:
                    signal_type, signal_energy = external_streams[node.node_id]
                    node.absorb_external_signal(signal_energy, signal_type)
        for node in self.nodes:
            node.predict_energy()
            node.clear_anxiety()
            node.replay_memories(self.nodes)

    def print_states(self):
        print(f"Time: {self.time}")
        print("Nodes:")
        for node in self.nodes:
            node.print_status()
        print("Capacitors:")
        for j, capacitor in enumerate(self.capacitors):
            print(f"Capacitor {j}: Position {capacitor.position}, Energy {round(capacitor.energy,2)}/{capacitor.capacity}")
        print()
