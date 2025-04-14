import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantum_ising_traffic import QuantumIsingTraffic
import folium

# Simulation Configuration
CONFIG = {
    # Location settings
    'place': "Bandar Utama, Petaling Jaya, Selangor, Malaysia",  # The area to simulate
    'network_type': 'drive',  # Type of road network: 'drive', 'walk', 'bike', etc.

    # Simulation parameters
    'simulation_steps': 2000,  # Number of steps in the quantum simulation
    'initial_temperature': 5.0,  # Starting temperature for simulated annealing
    'cooling_rate': 0.99,  # Rate at which temperature decreases (0-1, closer to 1 = slower cooling)

    # Road interaction parameters
    'interaction_strength': 1.0,  # Strength of interaction between connected roads
                                 # Higher values make connected roads more likely to have the same state

    # Road priority parameters
    'highway_field_strength': 1.0,  # External field strength for highways (higher = more likely to be open)
    'base_field_strength': 0.1,     # External field strength for regular roads
    'highway_identifier': 'LDP',    # Text to identify highways in road names

    # Quantum parameters
    'tunneling_strength': 0.2,  # Strength of quantum tunneling effect (higher = more tunneling)
    'entanglement_range': 3,    # Range of entanglement-like correlations between roads

    # Output settings
    'map_output_file': 'quantum_ising_routing_map.html',  # Name of the output HTML file
    'energy_plot_file': 'quantum_energy_history.png',     # Name of the energy history plot file

    # Road closure settings
    'closures': []  # List of road indices that should remain permanently closed (-1)
                   # These roads will be fixed in the closed state during simulation
                   # Example: [5, 10, 15] would keep roads at indices 5, 10, and 15 closed
}

# Load and prepare the road network
G = ox.graph_from_place(CONFIG['place'], network_type=CONFIG['network_type'])
G = ox.simplify_graph(G)

edges = list(G.edges(keys=True, data=True))
num_edges = len(edges)
edge_to_index = {edge[:3]: i for i, edge in enumerate(edges)}  # (u, v, key) → index

J = np.zeros((num_edges, num_edges))

# Connect segments sharing a node (simple heuristic)
for i, (u1, v1, k1, d1) in enumerate(edges):
    for j, (u2, v2, k2, d2) in enumerate(edges):
        if i == j:
            continue
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            J[i, j] = CONFIG['interaction_strength']

h = np.zeros(num_edges)
for i, (_, _, _, data) in enumerate(edges):
    if data.get('name') and CONFIG['highway_identifier'] in data['name']:
        h[i] = CONFIG['highway_field_strength']
    else:
        h[i] = CONFIG['base_field_strength']

# Create quantum model
model = QuantumIsingTraffic(
    num_nodes=num_edges,
    J_matrix=J,
    h_field=h,
    closures=CONFIG['closures'],
    tunneling_strength=CONFIG['tunneling_strength'],
    entanglement_range=CONFIG['entanglement_range']
)

# Run quantum simulation
spins = model.run(
    steps=CONFIG['simulation_steps'],
    T0=CONFIG['initial_temperature'],
    cooling=CONFIG['cooling_rate'],
    track_energy=True,
    track_interval=10  # Track energy more frequently for better visualization
)

# Get quantum state information
quantum_state = model.get_quantum_state()
amplitudes = quantum_state['amplitudes']
entanglement = quantum_state['entanglement']

# Create map visualization
m = ox.plot_graph_folium(G, popup_attribute='name', weight=1, color='gray')

# Add roads with quantum-inspired coloring
for i, (u, v, k, data) in enumerate(edges):
    if spins[i] == 1:  # Open road
        # Calculate "quantumness" based on amplitude
        quantumness = abs(amplitudes[i, 1])**2  # Probability of being open
        
        # Color based on quantumness (red = classical, blue = quantum)
        color = f'#{int(255*quantumness):02x}00{int(255*(1-quantumness)):02x}'
        
        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        folium.PolyLine(coords, color=color, weight=4).add_to(m)

# Save map
m.save(CONFIG['map_output_file'])

# Plot energy history
plt.figure(figsize=(10, 6))
plt.plot(model.get_energy_history())
plt.title('Quantum Energy History')
plt.xlabel('Simulation Step')
plt.ylabel('Energy')
plt.grid(True)
plt.savefig(CONFIG['energy_plot_file'])
plt.close()

print(f"Simulation complete. Results saved to {CONFIG['map_output_file']} and {CONFIG['energy_plot_file']}")
print(f"Final energy: {model.get_energy_history()[-1]}")
print(f"Number of open roads: {np.sum(spins == 1)}")
print(f"Number of closed roads: {np.sum(spins == 0)}") 