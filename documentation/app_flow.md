# App Flow

1. User selects location and simulation parameters from the sidebar.
2. Road network is fetched and simplified with OSMnx.
3. Each road segment becomes an Ising "spin":
   - Coupling matrix `J` is constructed based on adjacency.
   - Field vector `h` is biased toward highways (e.g., LDP).
4. User may manually input closures by edge index.
5. Upon running simulation:
   - `QuantumIsingTraffic` model is initialized and stepped.
   - Spin states evolve under simulated annealing + quantum terms.
   - Final states are visualized on a folium map.
   - Energy over time is plotted using `st.line_chart`.
