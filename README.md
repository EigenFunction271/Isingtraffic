# Ising Traffic Simulation

This project implements traffic simulations using both classical and quantum-inspired Ising models. It models road networks as a system of interacting spins, where each road segment can be either open (+1) or closed (-1). The simulations use simulated annealing to find optimal traffic patterns.

## Features

### Classical Implementation
- Road network representation using OSMnx
- Ising model-based traffic flow simulation
- Simulated annealing optimization
- Interactive visualization using Folium
- Configurable parameters for simulation
- Animation capabilities to visualize simulation progress

### Quantum-Inspired Implementation
- Quantum superposition states for roads
- Quantum tunneling effects
- Entanglement-like correlations between roads
- Enhanced visualization of quantum states
- Energy history tracking
- Animation capabilities to visualize quantum evolution

### Comparison Tools
- Side-by-side comparison of classical and quantum models
- Energy evolution visualization
- Interactive map comparisons
- Configurable simulation parameters
- Support for road closures and highway prioritization

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd IsingTraffic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Classical Simulation
```bash
python map_setup.py
```
This will generate `ising_routing_map.html` showing the classical simulation results.

1. Configure the simulation parameters in `map_setup.py`:
```python
CONFIG = {
    'place': "Your City, Country",  # Location to simulate
    'network_type': 'drive',        # Type of road network
    'simulation_steps': 2000,       # Number of simulation steps
    'initial_temperature': 5.0,     # Starting temperature for simulated annealing
    'cooling_rate': 0.99,           # Temperature cooling rate
    'interaction_strength': 1.0,    # Strength of road interactions
    'highway_field_strength': 1.0,  # Priority for highways
    'base_field_strength': 0.1,     # Base priority for regular roads
    'highway_identifier': 'LDP',    # Identifier for highways in road names
    'map_output_file': 'ising_routing_map.html',  # Output file name
    'closures': []                  # List of closed road indices
}
```

2. Run the classical simulation:
```bash
python map_setup.py
```

### Running Classical Animation
```bash
python animate_simulation.py
```
This will generate `ising_simulation.gif` showing the evolution of the classical simulation.

1. Configure the animation parameters in `animate_simulation.py`:
```python
CONFIG = {
    # Location settings
    'place': "Your City, Country",
    'network_type': 'drive',
    
    # Simulation parameters
    'simulation_steps': 2000,
    'initial_temperature': 5.0,
    'cooling_rate': 0.99,
    
    # Road interaction parameters
    'interaction_strength': 1.0,
    
    # Road priority parameters
    'highway_field_strength': 1.0,
    'base_field_strength': 0.1,
    'highway_identifier': 'LDP',
    
    # Animation settings
    'frames_per_second': 10,        # Number of frames per second in the GIF
    'frame_interval': 50,           # Take a snapshot every N steps
    'gif_output_file': 'ising_simulation.gif',  # Name of the output GIF file
    'temp_image_dir': 'temp_frames',  # Directory to store temporary frame images
    
    # Road closure settings
    'closures': []
}
```

### Quantum-Inspired Simulation

1. Configure the quantum simulation parameters in `quantum_map_setup.py`:
```python
CONFIG = {
    # Location settings
    'place': "Your City, Country",
    'network_type': 'drive',
    
    # Simulation parameters
    'simulation_steps': 2000,
    'initial_temperature': 5.0,
    'cooling_rate': 0.99,
    
    # Road interaction parameters
    'interaction_strength': 1.0,
    
    # Road priority parameters
    'highway_field_strength': 1.0,
    'base_field_strength': 0.1,
    'highway_identifier': 'LDP',
    
    # Quantum parameters
    'tunneling_strength': 0.2,  # Strength of quantum tunneling effect
    'entanglement_range': 3,    # Range of entanglement-like correlations
    
    # Output settings
    'map_output_file': 'quantum_ising_routing_map.html',
    'energy_plot_file': 'quantum_energy_history.png',
    
    # Road closure settings
    'closures': []
}
```

2. Run the quantum-inspired simulation:
```bash
python quantum_map_setup.py
```

3. Open the generated HTML file in a web browser to view the results.

### Running Quantum Animation
```bash
python animate_quantum_simulation.py
```
This will generate `quantum_ising_simulation.gif` showing the evolution of the quantum simulation.

1. Configure the quantum animation parameters in `animate_quantum_simulation.py`:
```python
CONFIG = {
    # Location settings
    'place': "Your City, Country",
    'network_type': 'drive',
    
    # Simulation parameters
    'simulation_steps': 2000,
    'initial_temperature': 10.0,  # Higher initial temperature for quantum effects
    'cooling_rate': 0.995,        # Slower cooling for quantum effects
    
    # Road interaction parameters
    'interaction_strength': 1.0,
    
    # Road priority parameters
    'highway_field_strength': 1.0,
    'base_field_strength': 0.1,
    'highway_identifier': 'LDP',
    
    # Quantum parameters
    'tunneling_strength': 0.3,  # Strength of quantum tunneling effect
    'entanglement_range': 4,    # Range of entanglement-like correlations
    
    # Animation settings
    'frames_per_second': 15,        # Number of frames per second in the GIF
    'frame_interval': 30,           # Take a snapshot every N steps
    'gif_output_file': 'quantum_ising_simulation.gif',  # Name of the output GIF file
    'temp_image_dir': 'temp_frames',  # Directory to store temporary frame images
    
    # Road closure settings
    'closures': []
}
```

### Comparing Classical and Quantum Simulations
```bash
python compare_simulations.py
```
This will generate comparison plots showing the differences between classical and quantum simulations.

1. Configure the comparison parameters in `compare_simulations.py`:
```python
CONFIG = {
    # Location settings
    'place': "Your City, Country",
    'network_type': 'drive',
    
    # Simulation parameters
    'simulation_steps': 2000,
    'initial_temperature': 5.0,
    'cooling_rate': 0.99,
    
    # Road interaction parameters
    'interaction_strength': 1.0,
    
    # Road priority parameters
    'highway_field_strength': 1.0,
    'base_field_strength': 0.1,
    'highway_identifier': 'LDP',
    
    # Quantum parameters
    'tunneling_strength': 0.2,
    'entanglement_range': 3,
    
    # Animation settings
    'frames_per_second': 10,
    'frame_interval': 50,
    'classical_gif': 'classical_simulation.gif',
    'quantum_gif': 'quantum_simulation.gif',
    'comparison_plot': 'simulation_comparison.png',
    'temp_image_dir': 'temp_frames',
    
    # Road closure settings
    'closures': []
}
```

## Mathematical Details

### Classical Ising Model

The classical Ising model represents each road as a spin with two possible states: +1 (open) or -1 (closed). The energy of the system is given by:

\[ E = -\frac{1}{2} \sum_{i,j} J_{ij} s_i s_j - \sum_i h_i s_i \]

where:
- \( s_i \) is the spin of road \( i \) (+1 or -1)
- \( J_{ij} \) is the interaction strength between roads \( i \) and \( j \)
- \( h_i \) is the external field affecting road \( i \)

The system evolves using simulated annealing, where the probability of accepting a state change is:

\[ P(\Delta E) = \min(1, e^{-\Delta E / T}) \]

where \( T \) is the temperature parameter that decreases over time.

### Quantum-Inspired Model

The quantum-inspired model extends the classical Ising model with quantum-like effects:

#### 1. Quantum Superposition

Each road is represented by a quantum state vector with complex amplitudes:

\[ |\psi_i\rangle = \alpha_i|0\rangle + \beta_i|1\rangle \]

where:
- \( |0\rangle \) represents a closed road
- \( |1\rangle \) represents an open road
- \( \alpha_i \) and \( \beta_i \) are complex amplitudes with \( |\alpha_i|^2 + |\beta_i|^2 = 1 \)

#### 2. Quantum Tunneling

The model includes a tunneling effect that allows roads to "tunnel" through energy barriers with probability:

\[ P_{\text{tunnel}} = \gamma \exp(-\Delta E / T) \]

where \( \gamma \) is the tunneling strength parameter.

#### 3. Entanglement-like Correlations

Roads can be correlated with each other through an entanglement range parameter \( r \). When a road's state changes, nearby roads within the entanglement range are affected according to their distance.

The entanglement measure is calculated as:

\[ E = (1/N)∑ᵢⱼ |⟨ψ|σᵢσⱼ|ψ⟩ - ⟨ψ|σᵢ|ψ⟩⟨ψ|σⱼ|ψ⟩| \]

#### 4. Measurement and Collapse

When measuring a road's state, the wavefunction collapses according to the Born rule:

\[ P(|0\rangle) = |\alpha_i|^2, \quad P(|1\rangle) = |\beta_i|^2 \]

## Configuration Parameters

### Common Parameters

- `place`: The location to simulate (city, country)
- `network_type`: Type of road network ('drive', 'walk', 'bike', etc.)
- `simulation_steps`: Number of steps in the simulation
- `initial_temperature`: Starting temperature for simulated annealing
- `cooling_rate`: Rate at which temperature decreases (0-1)
- `interaction_strength`: Strength of interactions between connected roads
- `highway_field_strength`: Priority given to highways
- `base_field_strength`: Base priority for regular roads
- `highway_identifier`: Text to identify highways in road names
- `map_output_file`: Name of the output HTML file
- `closures`: List of road indices that should remain closed

### Quantum-Specific Parameters

- `tunneling_strength`: Strength of quantum tunneling effect (higher = more tunneling)
- `entanglement_range`: Range of entanglement-like correlations between roads
- `energy_plot_file`: Name of the energy history plot file

### Animation Parameters

- `frames_per_second`: Number of frames per second in the GIF
- `frame_interval`: Take a snapshot every N steps
- `gif_output_file`: Name of the output GIF file
- `temp_image_dir`: Directory to store temporary frame images

## How It Works

1. The code downloads a real road network from OpenStreetMap
2. Each road segment is represented as a spin in the Ising model
3. Connected roads interact with each other
4. Highways are given higher priority through external fields
5. The system evolves using simulated annealing (classical) or quantum-inspired dynamics
6. The final configuration shows which roads should be open/closed
7. Results are visualized on an interactive map or as an animation

## Dependencies

- osmnx: OpenStreetMap data access
- networkx: Graph operations
- numpy: Numerical computations
- folium: Interactive map visualization
- pandas: Data manipulation
- matplotlib: Plotting
- scipy: Scientific computing (for quantum implementation)
- Pillow: Image processing (for GIF creation)

## License

This project is open source and available under the MIT License.

## Appendix: Quantum Model Implementation Details

### Mathematical Foundation

The quantum-inspired traffic model extends the classical Ising model with quantum mechanical concepts. Here's a detailed breakdown of the implementation:

#### 1. Quantum State Representation

Each road in the network is represented by a quantum state vector:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

where:
- |0⟩ represents a closed road
- |1⟩ represents an open road
- α and β are complex amplitudes satisfying |α|² + |β|² = 1
- The probability of measuring the road as open is |β|²
- The probability of measuring the road as closed is |α|²

The system is initialized in an equal superposition state:
```
|ψ₀⟩ = (1/√2)(|0⟩ + |1⟩)
```

#### 2. Energy Function

The total energy of the system includes both classical and quantum terms:

```
E = -∑ᵢⱼ Jᵢⱼ⟨ψ|σᵢσⱼ|ψ⟩ - ∑ᵢ hᵢ⟨ψ|σᵢ|ψ⟩ + quantum_corrections
```

where:
- Jᵢⱼ is the interaction strength between roads i and j
- hᵢ is the external field affecting road i
- σᵢ is the Pauli-Z operator for road i
- quantum_corrections include:
  1. Tunneling energy: -γ∑ᵢ ⟨ψ|σₓ|ψ⟩
  2. Entanglement energy: -Jₑ∑ᵢⱼₖ ⟨ψ|σᵢσⱼσₖ|ψ⟩

#### 3. Quantum Effects

##### a. Quantum Tunneling
Roads can "tunnel" through energy barriers with probability:
```
P = min(1, exp(-ΔE/T))
```
where:
- ΔE is the energy difference
- T is the temperature
- The tunneling strength γ controls how easily roads can tunnel

##### b. Entanglement-like Correlations
Roads within a certain range (entanglement_range) can be correlated beyond direct connections. The entanglement measure is:
```
E = (1/N)∑ᵢⱼ |⟨ψ|σᵢσⱼ|ψ⟩ - ⟨ψ|σᵢ|ψ⟩⟨ψ|σⱼ|ψ⟩|
```

##### c. Quantum Interference
Paths can interfere constructively or destructively based on their phases:
```
|ψ'⟩ = exp(-iHΔt)|ψ⟩
```
where H is the Hamiltonian of the system.

#### 4. Time Evolution

The quantum state evolves according to:
1. Select a road i
2. Calculate energy change ΔE
3. Apply quantum tunneling with probability P
4. Update quantum state with phase rotation:
   ```
   |ψ'⟩ = exp(iθ)|ψ⟩
   ```
5. Normalize the state
6. Measure to get classical configuration

#### 5. Measurement Process

The quantum state collapses to a classical state based on the Born rule:
- P(|1⟩) = |β|² (probability of measuring open)
- P(|0⟩) = |α|² (probability of measuring closed)

#### 6. Quantumness Measure

The "quantumness" of a road measures how far its state is from classical:
```
quantumness = 1 - |⟨ψ|σᵢ|ψ⟩|²
```
- quantumness = 0 for classical states (|0⟩ or |1⟩)
- quantumness = 1 for equal superposition ((|0⟩ + |1⟩)/√2)

#### 7. Implementation Parameters

Key parameters in the implementation:
- `tunneling_strength`: Controls quantum tunneling probability (default: 0.2-0.3)
- `entanglement_range`: Range of entanglement-like correlations (default: 3-4)
- `initial_temperature`: Starting temperature for simulated annealing (default: 5.0-10.0)
- `cooling_rate`: Rate at which temperature decreases (default: 0.99-0.995)

#### 8. Visualization

The quantum state is visualized through:
- Color gradients based on quantumness (red = classical, blue = quantum)
- Line thickness indicating road state (open/closed)
- Animation showing the evolution of the quantum state over time

This quantum-inspired implementation allows the traffic system to explore configurations that might be inaccessible to classical methods, potentially finding more optimal solutions through quantum tunneling and entanglement-like effects.

#### 9. Entanglement and State Space Evolution

The entanglement measure plays a crucial role in shaping the evolution of the traffic system's state space:

##### a. State Space Connectivity

In a classical Ising model, the state space is a discrete hypercube where each vertex represents a specific configuration of open/closed roads. Transitions between states occur by flipping individual spins, moving along the edges of this hypercube.

The entanglement measure introduces additional connectivity in the state space:
- Roads that are entangled can change state together, even if they are not directly connected in the road network
- This creates "shortcuts" through the state space, allowing the system to explore configurations that would require multiple sequential changes in the classical model
- The entanglement range parameter controls how many of these shortcuts are available

##### b. Collective Behavior

The entanglement measure promotes collective behavior among roads:
- When roads are highly entangled, they tend to evolve together toward similar states
- This creates clusters of roads that behave as a single unit, reducing the effective dimensionality of the state space
- The system can thus find solutions that maintain coherence across larger sections of the road network

##### c. Energy Landscape Modification

The entanglement measure modifies the energy landscape of the system:
- It introduces additional energy terms that depend on the correlation between roads
- These terms create valleys and ridges in the energy landscape that guide the system toward configurations with strong correlations
- The system can "slide" along these features, potentially avoiding local minima that would trap a classical system

##### d. Exploration vs. Exploitation

The entanglement measure affects the balance between exploration and exploitation:
- High entanglement promotes exploitation by encouraging roads to align with their neighbors
- Low entanglement allows for more independent exploration of different configurations
- The entanglement history tracks how this balance evolves during the simulation

##### e. Phase Transitions

The entanglement measure can lead to phase transitions in the system:
- As temperature decreases, the system may transition from a disordered phase with low entanglement to an ordered phase with high entanglement
- These transitions can be observed as sudden changes in the entanglement history
- The critical temperature at which these transitions occur depends on the entanglement range and strength

##### f. Practical Implications

From a traffic optimization perspective, the entanglement measure has several practical implications:
- It promotes the formation of coherent traffic patterns where connected roads tend to be either all open or all closed
- It helps maintain the integrity of major routes by keeping all segments of a route in the same state
- It can lead to more robust solutions that are less sensitive to local perturbations

The entanglement measure thus transforms the optimization process from a purely local search to one that can discover globally coherent solutions, potentially finding traffic patterns that would be difficult to discover using classical methods alone. 