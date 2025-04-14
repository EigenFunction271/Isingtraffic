import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ising_traffic import IsingTraffic
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
    'simulation_steps': 2000,  # Number of steps in the simulation
    'initial_temperature': 5.0,  # Starting temperature for simulated annealing
    'cooling_rate': 0.99,  # Rate at which temperature decreases (0-1, closer to 1 = slower cooling)

    # Road interaction parameters
    'interaction_strength': 1.0,  # Strength of interaction between connected roads

    # Road priority parameters
    'highway_field_strength': 1.0,  # External field strength for highways
    'base_field_strength': 0.1,     # External field strength for regular roads
    'highway_identifier': 'LDP',    # Text to identify highways in road names

    # Animation settings
    'frames_per_second': 10,        # Number of frames per second in the GIF
    'frame_interval': 50,           # Take a snapshot every N steps
    'gif_output_file': 'classical_ising_simulation.gif',  # Name of the output GIF file
    'temp_image_dir': 'temp_frames_classical',  # Directory to store temporary frame images

    # Road closure settings
    'closures': [3,5]  # List of road indices that should remain permanently closed (-1)
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

# Initialize the classical model
model = IsingTraffic(
    num_nodes=num_edges,
    J_matrix=J,
    h_field=h,
    closures=CONFIG['closures']
)

# Function to create a map frame
def create_map_frame(spins, step, temperature):
    m = ox.plot_graph_folium(G, popup_attribute='name', weight=1, color='gray')
    
    # Add roads with colors based on their state
    for i, (u, v, k, data) in enumerate(edges):
        if spins[i] == 1:  # Open road
            coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
            folium.PolyLine(coords, color='red', weight=4).add_to(m)
    
    # Add title with step and temperature
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 300px; height: 30px; 
                z-index:9999; font-size:14px; background-color: white; 
                padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
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
print("Running classical simulation and capturing frames...")
T = CONFIG['initial_temperature']
frame_files = []

for step in range(CONFIG['simulation_steps']):
    # Perform one step of the simulation
    model.step(T)
    
    # Take a snapshot at regular intervals
    if step % CONFIG['frame_interval'] == 0:
        print(f"Capturing frame at step {step} (temperature: {T:.4f})")
        # Get current states
        spins = model.spins
        
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
ax.set_title("Classical Ising Traffic Simulation")

# Function to update the plot for each frame
def update(frame):
    ax.clear()
    
    # Plot the road network
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1, alpha=0.5)
    
    # Get current states
    spins = model.spins
    
    # Draw open roads with red color
    open_edges = []
    for i, (u, v, k, _) in enumerate(edges):
        if spins[i] == 1:  # Open road
            open_edges.append((u, v, k))
    
    if open_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=open_edges, 
                             edge_color='red', width=2)
    
    # Set title with step and temperature
    step = frame * CONFIG['frame_interval']
    temperature = CONFIG['initial_temperature'] * (CONFIG['cooling_rate'] ** step)
    ax.set_title(f"Step: {step}, Temperature: {temperature:.4f}")
    
    # Set axis limits
    ax.set_xlim(min(x for x, _ in pos.values()) - 0.01, max(x for x, _ in pos.values()) + 0.01)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.01, max(y for _, y in pos.values()) + 0.01)
    
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