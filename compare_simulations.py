import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ising_traffic import IsingTraffic
from quantum_ising_traffic import QuantumIsingTraffic
import folium
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import os

# Simulation Configuration
CONFIG = {
    # Location settings
    'place': "Bandar Utama, Petaling Jaya, Selangor, Malaysia",
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

def setup_network():
    """Load and prepare the road network."""
    G = ox.graph_from_place(CONFIG['place'], network_type=CONFIG['network_type'])
    
    if not G.graph.get('simplified', False):
        G = ox.simplify_graph(G)
        print("Graph simplified")
    else:
        print("Graph was already simplified")
    
    edges = list(G.edges(keys=True, data=True))
    num_edges = len(edges)
    edge_to_index = {edge[:3]: i for i, edge in enumerate(edges)}
    
    # Create interaction matrix
    J = np.zeros((num_edges, num_edges))
    for i, (u1, v1, k1, d1) in enumerate(edges):
        for j, (u2, v2, k2, d2) in enumerate(edges):
            if i == j:
                continue
            if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                J[i, j] = CONFIG['interaction_strength']
    
    # Create external field
    h = np.zeros(num_edges)
    for i, (_, _, _, data) in enumerate(edges):
        if data.get('name') and CONFIG['highway_identifier'] in data['name']:
            h[i] = CONFIG['highway_field_strength']
        else:
            h[i] = CONFIG['base_field_strength']
    
    return G, edges, J, h

def run_classical_simulation(J, h):
    """Run the classical Ising model simulation."""
    print("Running classical simulation...")
    model = IsingTraffic(
        num_nodes=len(h),
        J_matrix=J,
        h_field=h,
        closures=CONFIG['closures']
    )
    
    spins = model.run(
        steps=CONFIG['simulation_steps'],
        T0=CONFIG['initial_temperature'],
        cooling=CONFIG['cooling_rate']
    )
    
    return model, spins

def run_quantum_simulation(J, h):
    """Run the quantum-inspired Ising model simulation."""
    print("Running quantum simulation...")
    model = QuantumIsingTraffic(
        num_nodes=len(h),
        J_matrix=J,
        h_field=h,
        closures=CONFIG['closures'],
        tunneling_strength=CONFIG['tunneling_strength'],
        entanglement_range=CONFIG['entanglement_range']
    )
    
    spins = model.run(
        steps=CONFIG['simulation_steps'],
        T0=CONFIG['initial_temperature'],
        cooling=CONFIG['cooling_rate']
    )
    
    return model, spins

def create_comparison_plot(classical_model, quantum_model, G, edges):
    """Create a comparison plot of both simulations."""
    print("Creating comparison plot...")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get final states
    classical_spins = classical_model.spins
    quantum_spins = quantum_model._get_all_states()
    quantum_state = quantum_model.get_quantum_state()
    quantum_amplitudes = quantum_state['amplitudes']
    
    # Plot classical results
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', width=1, alpha=0.5)
    
    open_edges = []
    for i, (u, v, k, _) in enumerate(edges):
        if classical_spins[i] == 1:
            open_edges.append((u, v, k))
    
    if open_edges:
        nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=open_edges, 
                             edge_color='red', width=2)
    
    ax1.set_title("Classical Ising Model")
    
    # Plot quantum results
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='gray', width=1, alpha=0.5)
    
    open_edges = []
    edge_colors = []
    for i, (u, v, k, _) in enumerate(edges):
        if quantum_spins[i] == 1:
            open_edges.append((u, v, k))
            quantumness = abs(quantum_amplitudes[i, 1])**2
            edge_colors.append((quantumness, 0, 1-quantumness))
    
    if open_edges:
        nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=open_edges, 
                             edge_color=edge_colors, width=2)
    
    ax2.set_title("Quantum-Inspired Ising Model")
    
    # Add energy history plots
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 6))
    
    classical_energy = classical_model.get_energy_history()
    quantum_energy = quantum_model.get_energy_history()
    
    steps = np.arange(len(classical_energy)) * CONFIG['frame_interval']
    ax3.plot(steps, classical_energy, 'r-', label='Energy')
    ax3.set_title("Classical Energy History")
    ax3.set_xlabel("Simulation Step")
    ax3.set_ylabel("Energy")
    ax3.grid(True)
    
    steps = np.arange(len(quantum_energy)) * CONFIG['frame_interval']
    ax4.plot(steps, quantum_energy, 'b-', label='Energy')
    ax4.set_title("Quantum Energy History")
    ax4.set_xlabel("Simulation Step")
    ax4.set_ylabel("Energy")
    ax4.grid(True)
    
    # Save plots
    fig.savefig(CONFIG['comparison_plot'])
    fig2.savefig('energy_comparison.png')
    plt.close('all')
    
    print(f"Comparison plots saved as {CONFIG['comparison_plot']} and energy_comparison.png")

def main():
    # Create temporary directory if it doesn't exist
    if not os.path.exists(CONFIG['temp_image_dir']):
        os.makedirs(CONFIG['temp_image_dir'])
    
    # Setup network
    G, edges, J, h = setup_network()
    
    # Run simulations
    classical_model, classical_spins = run_classical_simulation(J, h)
    quantum_model, quantum_spins = run_quantum_simulation(J, h)
    
    # Create comparison plots
    create_comparison_plot(classical_model, quantum_model, G, edges)
    
    # Clean up
    try:
        if os.path.exists(CONFIG['temp_image_dir']):
            for file in os.listdir(CONFIG['temp_image_dir']):
                try:
                    os.remove(os.path.join(CONFIG['temp_image_dir'], file))
                except PermissionError:
                    print(f"Warning: Could not remove file {file}")
            os.rmdir(CONFIG['temp_image_dir'])
    except PermissionError:
        print(f"Warning: Could not remove temporary directory {CONFIG['temp_image_dir']}")
        print("You may need to manually delete this directory later.")

if __name__ == "__main__":
    main() 