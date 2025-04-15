# Backend Structure

## Core Components

- `quantum_ising_traffic.py`: The quantum-inspired Ising traffic model.
  - Handles state evolution with tunneling and entanglement effects.
  - Accepts interaction matrix `J`, field `h`, closures, and simulation parameters.

- `app.py`: Streamlit frontend that:
  - Accepts user input for location, parameters, and closures.
  - Loads road network via OSMnx.
  - Computes `J` and `h` matrices.
  - Runs the simulation and visualizes results.

## Dependencies
- `osmnx`: for road network extraction
- `folium`: for map rendering
- `numpy`: numerical ops
- `streamlit`: UI layer
- `streamlit-folium`: for folium display integration
