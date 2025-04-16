import streamlit as st
import numpy as np
import osmnx as ox
import folium
from streamlit_folium import st_folium
from quantum_ising_traffic import QuantumIsingTraffic

# Add after imports, before other code
if "simulation_complete" not in st.session_state:
    st.session_state.simulation_complete = False
if "results" not in st.session_state:
    st.session_state.results = None

@st.cache_data
def load_graph(place):
    G = ox.graph_from_place(place, network_type='drive')
    if not G.graph.get('simplified', False):
        G = ox.simplify_graph(G)
    return G

# Move all initialization code into a function
def initialize_network(place):
    G = load_graph(place)
    edges = list(G.edges(keys=True, data=True))
    num_edges = len(edges)
    edge_to_index = {edge[:3]: i for i, edge in enumerate(edges)}
    
    # Compute J and h
    J = np.zeros((num_edges, num_edges))
    h = np.zeros(num_edges)
    
    for i, (u1, v1, k1, d1) in enumerate(edges):
        for j, (u2, v2, k2, d2) in enumerate(edges):
            if i != j and (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2):
                J[i, j] = 1.0
        name = d1.get("name", "")
        h[i] = 1.0 if "LDP" in name else 0.1
    
    return G, edges, num_edges, edge_to_index, J, h

# Main app code
st.set_page_config(page_title="Quantum Traffic Playground", layout="wide")
st.title("🧠 Quantum-Inspired Traffic Playground")
st.markdown("""
Interactively explore rerouting in traffic systems using a quantum-inspired Ising model. 
You can simulate road closures and observe how the network adapts.
""")

# --- Sidebar config ---
st.sidebar.header("⚙️ Simulation Settings")
place = st.sidebar.text_input("Location", "Bandar Utama, Petaling Jaya, Selangor, Malaysia")
steps = st.sidebar.slider("Simulation Steps", 100, 5000, 1000, step=100)
temp = st.sidebar.slider("Initial Temperature", 1.0, 100.0, 30.0)
cooling = st.sidebar.slider("Cooling Rate", 0.90, 0.999, 0.995, step=0.001)
tunnel_strength = st.sidebar.slider("Tunneling Strength", 0.0, 1.0, 0.3)
entanglement = st.sidebar.slider("Entanglement Range", 0, 10, 3)

st.sidebar.markdown("---")
st.sidebar.header("🚧 Closures")
selected_closures = st.sidebar.text_area("Comma-separated indices of closed roads", "")

# Add toggle for quantum vs classical simulation
st.sidebar.markdown("---")
simulation_mode = st.sidebar.radio(
    "🔄 Simulation Mode",
    ["Quantum", "Classical"],
    help="Choose between quantum-inspired or classical simulation"
)

# Add quantumness histogram
if simulation_mode == "Quantum":
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Quantum Metrics")
    show_histogram = st.sidebar.checkbox("Show Quantumness Distribution")

# Initialize network once
with st.spinner("Loading road network..."):
    G, edges, num_edges, edge_to_index, J, h = initialize_network(place)

# Add interactive map for closure selection
st.info("💡 Click on roads in the map to select closures, or enter indices manually below.")
if selected_closures:
    st.caption(f"Currently selected closures: {selected_closures}")

# --- Closure parsing ---
closures = []
if selected_closures.strip():
    try:
        closures = [int(x.strip()) for x in selected_closures.split(",")]
    except:
        st.sidebar.error("Invalid closure indices. Use comma-separated integers.")

# Initialize appropriate model based on mode
if simulation_mode == "Classical":
    from ising_traffic import IsingTraffic
    model = IsingTraffic(
        num_nodes=num_edges,
        J_matrix=J,
        h_field=h,
        closures=closures
    )
else:
    model = QuantumIsingTraffic(
        num_nodes=num_edges,
        J_matrix=J,
        h_field=h,
        closures=closures,
        tunneling_strength=tunnel_strength,
        entanglement_range=entanglement
    )

# Update run button text based on mode
run_button_text = "🧪 Run Quantum Simulation" if simulation_mode == "Quantum" else "▶️ Run Classical Simulation"
if st.button(run_button_text):
    with st.spinner("Simulating..."):
        final_spins = model.run(steps=steps, T0=temp, cooling=cooling)
        
        # Store results in session state
        st.session_state.results = {
            'final_spins': final_spins,
            'mode': simulation_mode,
            'energy': model.get_energy_history()
        }
        
        if simulation_mode == "Quantum":
            quantum_state = model.get_quantum_state()
            st.session_state.results['amplitudes'] = quantum_state['amplitudes']
            
        st.session_state.simulation_complete = True
        st.rerun()

# After the button code, add:
if st.session_state.simulation_complete and st.session_state.results is not None:
    results = st.session_state.results
    
    if results['mode'] == "Quantum" and 'amplitudes' in results and show_histogram:
        quantumness = [abs(results['amplitudes'][i, 1])**2 for i in range(num_edges)]
        st.subheader("📊 Quantumness Distribution")
        st.caption("Shows how quantum-like each road's state is (0 = classical, 1 = quantum)")
        st.histogram_chart(quantumness)

    # Create visualization
    center_lat = G.nodes[list(G.nodes())[0]]['y']
    center_lng = G.nodes[list(G.nodes())[0]]['x']
    m = folium.Map(location=[center_lat, center_lng], zoom_start=15)

    # Plot the base network
    for u, v, data in G.edges(data=True):
        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        folium.PolyLine(
            coords,
            color='gray',
            weight=1,
            popup=data.get('name', ''),
            opacity=0.8
        ).add_to(m)

    for i, (u, v, k, data) in enumerate(edges):
        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        if results['final_spins'][i] == 1:
            if results['mode'] == "Quantum":
                quantumness = abs(results['amplitudes'][i, 1])**2
                color = f'#{int(255*quantumness):02x}00{int(255*(1-quantumness)):02x}'
            else:
                color = '#ff0000'  # Red for classical mode
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)
        else:
            folium.PolyLine(coords, color='#404040', weight=2, opacity=0.4).add_to(m)

    st.subheader("🎮 Simulation Result")
    st_folium(m, width=900, height=600, key=f"simulation_map_{st.session_state.simulation_complete}")
    
    st.subheader("📉 Energy Over Time")
    st.line_chart(results['energy'])
    
    # Success message
    if results['mode'] == "Quantum":
        avg_quantumness = np.mean([abs(results['amplitudes'][i, 1])**2 for i in range(num_edges)])
        st.success(f"""
        ✨ Quantum simulation complete!
        - Average quantumness: {avg_quantumness:.2f}
        - Open roads: {np.sum(results['final_spins'] == 1)}
        - Closed roads: {np.sum(results['final_spins'] == -1)}
        """)
    else:
        st.success(f"""
        ✅ Classical simulation complete!
        - Open roads: {np.sum(results['final_spins'] == 1)}
        - Closed roads: {np.sum(results['final_spins'] == -1)}
        """)

else:
    mode_text = "quantum" if simulation_mode == "Quantum" else "classical"
    st.info(f"🎮 Set parameters and click \"{run_button_text}\" to begin {mode_text} simulation.")

