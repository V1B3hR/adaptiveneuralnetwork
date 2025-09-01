# **AliveLoopNode Class Documentation**

## **Overview**

The `AliveLoopNode` class is designed to simulate the behavior of a node in a dynamic, interactive environment. It incorporates concepts such as energy management, circadian rhythms, memory systems, and social interactions. The class is highly modular and supports advanced features like neuromodulation, learning, and memory consolidation.

## **Class Features**

- **Energy Management**: Nodes consume energy for movement, communication, and other tasks. They can interact with capacitors to transfer energy.
- **Circadian Rhythms**: Nodes follow a 24-hour cycle that influences their phases (e.g., active, sleep, inspired).
- **Memory Systems**: Includes short-term, long-term, and working memory for storing and sharing information.
- **Social Interaction**: Nodes can communicate and share memories with trusted neighbors.
- **Adaptation and Learning**: Nodes adapt based on past experiences using mechanisms like Long-Term Potentiation (LTP) and Long-Term Depression (LTD).

---

## **Class: AliveLoopNode**

### **Constructor**
```python
def __init__(self, position, velocity, initial_energy=0.0, field_strength=1.0, node_id=0):
```

#### **Parameters**
- `position` (tuple): Initial position of the node (x, y).
- `velocity` (tuple): Initial velocity of the node (vx, vy).
- `initial_energy` (float): Initial energy level of the node. Default is `0.0`.
- `field_strength` (float): Field strength of the node. Default is `1.0`.
- `node_id` (int): Unique identifier for the node. Default is `0`.

#### **Attributes**
- `position` (np.array): Current position of the node.
- `velocity` (np.array): Current velocity of the node.
- `energy` (float): Current energy level of the node.
- `phase` (str): Current phase of the node (e.g., "active", "sleep").
- `memory` (deque): Long-term memory storage.
- `working_memory` (deque): Short-term memory storage.
- `trust_network` (dict): Trust levels for interacting with other nodes.

---

### **Method: step_phase**
```python
def step_phase(self, current_time):
```

#### **Description**
Manages the node's phase transitions based on circadian rhythms, sleep debt, and neuromodulator levels.

#### **Parameters**
- `current_time` (int): Current time in hours (0-24).

#### **Behavior**
- Calculates circadian phase using the current time.
- Determines whether the node enters sleep, active, or inspired phases.
- Updates the sleep debt based on the current phase.

---

### **Method: move**
```python
def move(self):
```

#### **Description**
Handles the node's movement and energy consumption.

#### **Behavior**
- Adjusts velocity based on the node's attention focus and phase.
- Ensures the velocity does not exceed the maximum speed.
- Updates the node's position and reduces energy based on movement.

---

### **Method: interact_with_capacitor**
```python
def interact_with_capacitor(self, capacitor, threshold=0.5):
```

#### **Description**
Manages energy transfer between the node and a capacitor.

#### **Parameters**
- `capacitor` (object): The capacitor object to interact with.
- `threshold` (float): Minimum distance for interaction. Default is `0.5`.

#### **Behavior**
- Calculates the distance between the node and the capacitor.
- Transfers energy based on the distance and energy levels of both entities.

---

### **Method: replay_memories**
```python
def replay_memories(self, nodes):
```

#### **Description**
Replays and shares memories with trusted nodes during sleep.

#### **Parameters**
- `nodes` (list): List of neighboring nodes.

#### **Behavior**
- Consolidates memories into long-term storage.
- Shares the most important memories with trusted neighboring nodes.

---

### **Method: _consolidate_memories**
```python
def _consolidate_memories(self):
```

#### **Description**
Consolidates working memory into long-term memory during sleep.

#### **Behavior**
- Moves memories with high importance into long-term storage.
- Ages and discards low-importance memories during deep sleep.

---

### **Method: adapt_and_learn**
```python
def adapt_and_learn(self):
```

#### **Description**
Adjusts the node's learning rate and energy efficiency based on past experiences.

#### **Behavior**
- Analyzes recent rewards to determine performance.
- Adjusts learning rate using LTP and LTD mechanisms.

---

## **Usage Examples**

### **Basic Node Initialization**
```python
from core.alive_node import AliveLoopNode

# Initialize a node at position (0, 0) with velocity (1, 1)
node = AliveLoopNode(position=(0, 0), velocity=(1, 1), initial_energy=10.0)
```

---

### **Simulating Node Behavior**
```python
# Step through phases of the day
for hour in range(24):
    node.step_phase(current_time=hour)
    node.move()
```

---

### **Interacting with a Capacitor**
```python
class MockCapacitor:
    def __init__(self, position, energy, capacity):
        self.position = np.array(position, dtype=float)
        self.energy = energy
        self.capacity = capacity

# Create a capacitor
capacitor = MockCapacitor(position=(5, 5), energy=15, capacity=20)

# Node interacts with the capacitor
node.interact_with_capacitor(capacitor=capacitor)
```

---

### **Sharing Memories**
```python
# Simulate neighboring nodes
neighboring_nodes = [AliveLoopNode(position=(2, 2), velocity=(0, 0), node_id=2)]

# Node replays memories with neighbors
node.replay_memories(nodes=neighboring_nodes)
```

---

## **Notes**
- Ensure that the node's `position` and `velocity` attributes are updated regularly to reflect interactions.
- Use the `trust_network` attribute to manage interactions with neighboring nodes effectively.
- The `memory` attribute is capped at 1000 entries to optimize performance.

---
