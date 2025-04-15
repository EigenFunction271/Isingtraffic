# Frontend Guidelines

## Map Interface
- Use `folium.PolyLine` to represent road segments.
- Color logic:
  - Closed roads: muted grey
  - Open roads: gradient red-blue by quantum probability

## Controls
- Place all parameters in the sidebar for clarity.
- Use `st.sidebar.slider()` for numeric inputs (steps, temp, etc.)
- Road closures input is comma-separated index list for now.

## Interactivity
- Keep the "Run Simulation" button clearly separate.
- After results are generated:
  - Show map (`st_folium`)
  - Show energy plot (`st.line_chart`)
  - Provide success message

## Accessibility
- Limit map width to 900px for layout balance
- Use consistent emojis/icons for headers
