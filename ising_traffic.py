import numpy as np
import random
from typing import List, Optional

class IsingTraffic:
    def __init__(self, num_nodes: int, J_matrix: np.ndarray, h_field: np.ndarray, closures: Optional[List[int]] = None):
        """
        Initialize the Ising Traffic model.
        
        Args:
            num_nodes: Number of road segments
            J_matrix: Interaction matrix between road segments
            h_field: External field affecting each road segment
            closures: List of indices of permanently closed roads
                     These roads will be fixed in the closed state (-1) during the simulation
                     and cannot be opened by the algorithm
        """
        # Input validation
        if not isinstance(J_matrix, np.ndarray) or J_matrix.shape != (num_nodes, num_nodes):
            raise ValueError(f"J_matrix must be a numpy array of shape ({num_nodes}, {num_nodes})")
        if not isinstance(h_field, np.ndarray) or len(h_field) != num_nodes:
            raise ValueError(f"h_field must be a numpy array of length {num_nodes}")
        
        self.N = num_nodes
        self.J = J_matrix
        self.h = h_field
        self.closures = [] if closures is None else list(closures)
        
        # Validate closures
        if not all(0 <= i < num_nodes for i in self.closures):
            raise ValueError("All closure indices must be between 0 and num_nodes-1")
        
        # Initialize spins (all roads open by default)
        self.spins = np.ones(self.N)
        for i in self.closures:
            self.spins[i] = -1  # permanently closed roads
        
        # Initialize energy tracking
        self.energy_history = []
        self.current_energy = self.energy()

    def energy(self) -> float:
        """
        Calculate the total energy of the system.
        Closed roads (in closures) don't contribute to the energy.
        """
        # Create a mask for active roads
        active_mask = np.ones_like(self.spins)
        active_mask[self.closures] = 0  # Mask out closed roads
        
        # Calculate interaction energy only for active roads
        interaction = -0.5 * np.sum(
            self.J * np.outer(self.spins * active_mask, self.spins * active_mask)
        )
        
        # Calculate field energy
        field = -np.sum(self.h * self.spins * active_mask)
        
        return interaction + field

    def delta_energy(self, i: int) -> float:
        """
        Calculate the energy change if spin i is flipped.
        Returns 0 for closed roads.
        """
        if i in self.closures:
            return 0  # don't flip closed roads
        dE = 2 * self.spins[i] * (np.sum(self.J[i] * self.spins) + self.h[i])
        return dE

    def step(self, T: float) -> None:
        """
        Perform one step of the simulation at temperature T.
        Closed roads are never selected for flipping.
        """
        i = random.randint(0, self.N - 1)
        if i in self.closures:
            return  # Skip closed roads
        
        dE = self.delta_energy(i)
        if dE < 0 or random.random() < np.exp(-dE / T):
            self.spins[i] *= -1
            self.current_energy += dE

    def run(self, steps: int = 1000, T0: float = 5.0, cooling: float = 0.99, 
            track_energy: bool = True, track_interval: int = 100) -> np.ndarray:
        """
        Run the simulation with specified parameters.
        
        Args:
            steps: Number of simulation steps
            T0: Initial temperature
            cooling: Temperature cooling rate
            track_energy: Whether to track energy history
            track_interval: How often to record energy (in steps)
            
        Returns:
            Final spin configuration
        """
        T = T0
        self.energy_history = []
        
        for step in range(steps):
            self.step(T)
            
            # Track energy if requested
            if track_energy and step % track_interval == 0:
                self.energy_history.append(self.current_energy)
            
            T *= cooling
        
        return self.spins

    def get_energy_history(self) -> List[float]:
        """Return the history of energy values during the simulation."""
        return self.energy_history


if __name__ == "__main__":
    # Example usage
    N = 10  # number of roads
    J = np.random.rand(N, N) * 0.5
    J = (J + J.T) / 2  # make symmetric
    np.fill_diagonal(J, 0)

    h = np.random.randn(N) * 0.2  # some roads more attractive
    closures = [3, 7]  # pretend these roads are closed

    model = IsingTraffic(num_nodes=N, J_matrix=J, h_field=h, closures=closures)
    final_state = model.run(track_energy=True)
    print("Final configuration:", final_state)
    print("Energy history:", model.get_energy_history())
