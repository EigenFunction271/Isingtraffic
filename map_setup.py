import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ising_traffic import IsingTraffic
import folium

# Simulation Configuration
CONFIG = {
    # Location settings
    'place': "Bandar Utama, Petaling Jaya, Selangor, Malaysia",  # The area to simulate
    'network_type': 'drive',  # Type of road network: 'drive', 'walk', 'bike', etc.

    # Simulation parameters
    'simulation_steps': 2000,  # Number of steps in the simulated annealing process
    'initial_temperature': 5.0,  # Starting temperature for simulated annealing (higher = more random)
    'cooling_rate': 0.99,  # Rate at which temperature decreases (0-1, closer to 1 = slower cooling)

    # Road interaction parameters
    'interaction_strength': 1.0,  # Strength of interaction between connected roads
                                 # Higher values make connected roads more likely to have the same state

    # Road priority parameters
    'highway_field_strength': 1.0,  # External field strength for highways (higher = more likely to be open)
    'base_field_strength': 0.1,     # External field strength for regular roads
    'highway_identifier': 'LDP',    # Text to identify highways in road names

    # Output settings
    'map_output_file': 'ising_routing_map.html',  # Name of the output HTML file

    # Road closure settings
    'closures': []  # List of road indices that should remain permanently closed (-1)
                   # To close specific roads:
                   # 1. First run the simulation without closures to see the road indices
                   # 2. Look at the road names in the generated map
                   # 3. Add the indices of roads you want to close to this list
                   # Example: [5, 10, 15] would keep roads at indices 5, 10, and 15 closed
                   # Note: Road indices correspond to the order in which they appear in the network
                   # You can find road indices by:
                   # 1. Running the simulation once
                   # 2. Opening the generated HTML map
                   # 3. Hovering over roads to see their names and indices
                   # 4. Adding the desired indices to this list
}

# Load and prepare the road network
G = ox.graph_from_place(CONFIG['place'], network_type=CONFIG['network_type'])

# Check if the graph needs simplification
if not G.graph.get('simplified', False):
    G = ox.simplify_graph(G)
    print("Graph simplified")
else:
    print("Graph was already simplified")

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

model = IsingTraffic(
    num_nodes=num_edges,
    J_matrix=J,
    h_field=h,
    closures=CONFIG['closures']  # These roads will be fixed in the closed state during simulation
)

spins = model.run(
    steps=CONFIG['simulation_steps'],
    T0=CONFIG['initial_temperature'],
    cooling=CONFIG['cooling_rate']
)

m = ox.plot_graph_folium(G, popup_attribute='name', weight=1, color='gray')

for i, (u, v, k, data) in enumerate(edges):
    if spins[i] == 1:
        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        folium.PolyLine(coords, color='red', weight=4).add_to(m)

m.save(CONFIG['map_output_file'])