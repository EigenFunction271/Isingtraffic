import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from quantum_ising_traffic import QuantumIsingTraffic
import folium
from PIL import Image
import io
import os
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Simulation Configuration
CONFIG = {
    # Location settings
    'place': "Bandar Utama, Petaling Jaya, Selangor, Malaysia",  # The area to simulate
    'network_type': 'drive',  # Type of road network: 'drive', 'walk', 'bike', etc.

    # Simulation parameters
    'simulation_steps': 2500,  # Number of steps in the quantum simulation
    'initial_temperature': 30.0,  # Starting temperature for simulated annealing
    'cooling_rate': 0.995,  # Rate at which temperature decreases (0-1, closer to 1 = slower cooling)

    # Road interaction parameters
    'interaction_strength': 1.0,  # Strength of interaction between connected roads

    # Road priority parameters
    'highway_field_strength': 1.0,  # External field strength for highways
    'base_field_strength': 0.1,     # External field strength for regular roads
    'highway_identifier': 'LDP',    # Text to identify highways in road names

    # Quantum parameters
    'tunneling_strength': 0.3,  # Strength of quantum tunneling effect
    'entanglement_range': 4,    # Range of entanglement-like correlations

    # Animation settings
    'frames_per_second': 15,        # Number of frames per second in the GIF
    'frame_interval': 30,           # Take a snapshot every N steps
    'gif_output_file': 'quantum_ising_simulation.gif',  # Name of the output GIF file
    'temp_image_dir': 'temp_frames',  # Directory to store temporary frame images

    # Road closure settings
    'closures': []  # List of road indices that should remain permanently closed (-1)
}

# Create temporary directory for frames if it doesn't exist
if not os.path.exists(CONFIG['temp_image_dir']):
    os.makedirs(CONFIG['temp_image_dir'])

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

# Initialize the quantum model
model = QuantumIsingTraffic(
    num_nodes=num_edges,
    J_matrix=J,
    h_field=h,
    closures=CONFIG['closures'],
    tunneling_strength=CONFIG['tunneling_strength'],
    entanglement_range=CONFIG['entanglement_range']
)

# Function to create a map frame
def create_map_frame(spins, step, temperature):
    m = ox.plot_graph_folium(G, popup_attribute='name', weight=1, color='gray')
    
    # Get quantum state information
    quantum_state = model.get_quantum_state()
    amplitudes = quantum_state['amplitudes']
    
    # Add roads with colors based on their state
    for i, (u, v, k, data) in enumerate(edges):
        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        
        if spins[i] == 1:  # Open road
            # Calculate quantumness based on amplitude
            quantumness = abs(amplitudes[i, 1])**2  # Probability of being open
            # Color based on quantumness (red = classical, blue = quantum)
            color = f'#{int(255*quantumness):02x}00{int(255*(1-quantumness)):02x}'
            folium.PolyLine(coords, color=color, weight=2, opacity=0.8).add_to(m)
        else:  # Closed road
            folium.PolyLine(coords, color='#404040', weight=2, opacity=0.5).add_to(m)
    
    # Add title with step and temperature
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 40px; 
                z-index:9999; font-size:16px; font-weight: bold;
                background-color: rgba(255, 255, 255, 0.9); 
                padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.3);">
        Step: {step}, Temperature: {temperature:.4f}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

# Function to save a map frame as an image
def save_frame_as_image(frame_map, frame_number):
    # Save the map as HTML
    temp_html = f"{CONFIG['temp_image_dir']}/frame_{frame_number:04d}.html"
    frame_map.save(temp_html)
    
    # Use Selenium to render the HTML and capture as image
    # This is a simplified approach - in a real implementation, you'd use Selenium
    # For now, we'll just save the HTML files and provide instructions
    
    return temp_html

# Run the simulation and capture frames
print("Running quantum simulation and capturing frames...")
T = CONFIG['initial_temperature']
frame_files = []

for step in range(CONFIG['simulation_steps']):
    # Perform one step of the simulation
    model.step(T)
    
    # Take a snapshot at regular intervals
    if step % CONFIG['frame_interval'] == 0:
        print(f"Capturing frame at step {step} (temperature: {T:.4f})")
        # Get current states
        spins = model.get_current_spins()
        
        frame_map = create_map_frame(spins, step, T)
        frame_file = save_frame_as_image(frame_map, step // CONFIG['frame_interval'])
        frame_files.append(frame_file)
    
    # Update temperature
    T *= CONFIG['cooling_rate']

print(f"Simulation complete. {len(frame_files)} frames captured.")

# Create a simple animation using matplotlib (alternative approach)
print("Creating animation using matplotlib...")

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Quantum Ising Traffic Simulation")

# Function to update the plot for each frame
def update(frame):
    ax.clear()
    
    # Plot the road network with a dark background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    # Get quantum state information
    quantum_state = model.get_quantum_state()
    amplitudes = quantum_state['amplitudes']
    
    # Plot the road network
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    
    # Draw roads with colors based on their state
    for i, (u, v, k, _) in enumerate(edges):
        if spins[i] == 1:  # Open road
            # Calculate quantumness based on amplitude
            quantumness = abs(amplitudes[i, 1])**2  # Probability of being open
            # Color based on quantumness (red = classical, blue = quantum)
            color = (quantumness, 0, 1-quantumness)
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v, k)],
                                 edge_color=color, width=4)
        else:  # Closed road
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v, k)],
                                 edge_color='#404040', width=1, alpha=0.5)
    
    # Calculate current step and temperature
    step = frame * CONFIG['frame_interval']
    temperature = CONFIG['initial_temperature'] * (CONFIG['cooling_rate'] ** step)
    
    # Add a text box with step and temperature information
    info_text = f'Step: {step}\nTemp: {temperature:.4f}'
    text_box = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=text_box, color='black', fontweight='bold')
    
    # Add a legend for quantum states
    quantum_text = 'Color Scale:\nRed = Classical\nBlue = Quantum'
    quantum_box = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.98, quantum_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=quantum_box, color='black')
    
    # Set axis limits with some padding
    ax.set_xlim(min(x for x, _ in pos.values()) - 0.01, max(x for x, _ in pos.values()) + 0.01)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.01, max(y for _, y in pos.values()) + 0.01)
    
    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax,

# Create the animation
anim = FuncAnimation(
    fig, 
    update, 
    frames=len(frame_files),
    interval=1000/CONFIG['frames_per_second'],
    blit=False
)

# Save the animation as a GIF
anim.save(CONFIG['gif_output_file'], writer='pillow', fps=CONFIG['frames_per_second'])
print(f"Animation saved as {CONFIG['gif_output_file']}")

# Clean up temporary files
print("Cleaning up temporary files...")
for file in frame_files:
    try:
        if os.path.exists(file):
            os.remove(file)
    except PermissionError:
        print(f"Warning: Could not remove temporary file {file}")

try:
    if os.path.exists(CONFIG['temp_image_dir']):
        # Try to remove all files in the directory first
        for file in os.listdir(CONFIG['temp_image_dir']):
            try:
                os.remove(os.path.join(CONFIG['temp_image_dir'], file))
            except PermissionError:
                print(f"Warning: Could not remove file {file}")
        
        # Then try to remove the directory
        os.rmdir(CONFIG['temp_image_dir'])
except PermissionError:
    print(f"Warning: Could not remove temporary directory {CONFIG['temp_image_dir']}")
    print("You may need to manually delete this directory later.")

print("Done!") 