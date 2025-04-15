import streamlit as st
import numpy as np
import osmnx as ox
import folium
from streamlit_folium import st_folium
from quantum_ising_traffic import QuantumIsingTraffic

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

# --- Load road network ---
st.subheader(f"📍 Road Network: {place}")
with st.spinner("Loading road network..."):
    G = ox.graph_from_place(place, network_type='drive')
    G = ox.simplify_graph(G)
    edges = list(G.edges(keys=True, data=True))
    num_edges = len(edges)
    edge_to_index = {edge[:3]: i for i, edge in enumerate(edges)}

# --- Compute J and h ---
J = np.zeros((num_edges, num_edges))
h = np.zeros(num_edges)

for i, (u1, v1, k1, d1) in enumerate(edges):
    for j, (u2, v2, k2, d2) in enumerate(edges):
        if i != j and (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2):
            J[i, j] = 1.0
    name = d1.get("name", "")
    h[i] = 1.0 if "LDP" in name else 0.1

# --- Closure parsing ---
closures = []
if selected_closures.strip():
    try:
        closures = [int(x.strip()) for x in selected_closures.split(",")]
    except:
        st.sidebar.error("Invalid closure indices. Use comma-separated integers.")

# --- Initialize model ---
model = QuantumIsingTraffic(
    num_nodes=num_edges,
    J_matrix=J,
    h_field=h,
    closures=closures,
    tunneling_strength=tunnel_strength,
    entanglement_range=entanglement
)

# --- Run simulation ---
if st.button("🧪 Run Quantum Simulation"):
    with st.spinner("Simulating..."):
        final_spins = model.run(steps=steps, T0=temp, cooling=cooling)
        amplitudes = model.get_quantum_state()['amplitudes']

    # --- Visualize result ---
    m = ox.plot_graph_folium(G, popup_attribute='name', weight=1, color='gray')

    for i, (u, v, k, data) in enumerate(edges):
        coords = [(G.nodes[u]['y'], G.nodes[u]['x']), (G.nodes[v]['y'], G.nodes[v]['x'])]
        if final_spins[i] == 1:
            quantumness = abs(amplitudes[i, 1])**2
            color = f'#{int(255*quantumness):02x}00{int(255*(1-quantumness)):02x}'
            folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(m)
        else:
            folium.PolyLine(coords, color='#404040', weight=2, opacity=0.4).add_to(m)

    st.subheader("🗺️ Simulation Result")
    st_folium(m, width=900, height=600)

    # --- Plot energy curve ---
    energy = model.get_energy_history()
    st.subheader("📉 Energy Over Time")
    st.line_chart(energy)

    st.success("Simulation complete.")
else:
    st.info("Set parameters and click \"Run Quantum Simulation\" to begin.")
