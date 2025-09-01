import random

class TunedAdaptiveFieldNetwork:
    def __init__(self, nodes, capacitors):
        self.nodes = nodes
        self.capacitors = capacitors
        self.time = 0

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
        # Absorb external signals
        if external_streams:
            for node in self.nodes:
                if node.node_id in external_streams:
                    signal_type, signal_energy = external_streams[node.node_id]
                    node.absorb_external_signal(signal_energy, signal_type)
        elif self.time % 10 == 0:
            for node in self.nodes:
                signal_type = random.choice(["human", "AI", "world"])
                signal_energy = random.uniform(1, 5)
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
