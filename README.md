# Ising Traffic Simulation

This project implements traffic simulations using both classical and quantum-inspired Ising models. It models road networks as a system of interacting spins, where each road segment can be either open (+1) or closed (-1). The simulations use simulated annealing to find optimal traffic patterns.

## Features

### Classical Implementation
- Road network representation using OSMnx
- Ising model-based traffic flow simulation
- Simulated annealing optimization
- Interactive visualization using Folium
- Configurable parameters for simulation

### Quantum-Inspired Implementation
- Quantum superposition states for roads
- Quantum tunneling effects
- Entanglement-like correlations between roads
- Enhanced visualization of quantum states
- Energy history tracking

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

Roads can be correlated with each other through an entanglement matrix \( E_{ij} \):

\[ E_{ij} = \frac{1}{1 + |i-j|/r} \]

where \( r \) is the entanglement range parameter.

When a road's state changes, entangled roads are affected according to:

\[ |\psi_j\rangle \rightarrow c_{ij} \cdot |\psi_i\rangle \]

where \( c_{ij} \) is the correlation strength from the entanglement matrix.

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

## How It Works

1. The code downloads a real road network from OpenStreetMap
2. Each road segment is represented as a spin in the Ising model
3. Connected roads interact with each other
4. Highways are given higher priority through external fields
5. The system evolves using simulated annealing (classical) or quantum-inspired dynamics
6. The final configuration shows which roads should be open/closed
7. Results are visualized on an interactive map

## Dependencies

- osmnx: OpenStreetMap data access
- networkx: Graph operations
- numpy: Numerical computations
- folium: Interactive map visualization
- pandas: Data manipulation
- matplotlib: Plotting
- scipy: Scientific computing (for quantum implementation)

## License

This project is open source and available under the MIT License. 