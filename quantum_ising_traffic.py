import numpy as np
import random
from typing import List, Optional, Tuple, Dict
import cmath
from scipy.linalg import expm

class QuantumIsingTraffic:
    """
    A quantum-inspired version of the Ising Traffic model.
    This implementation includes:
    - Quantum superposition states
    - Quantum tunneling effects
    - Entanglement-like correlations
    - Quantum interference patterns
    """
    
    def __init__(self, num_nodes: int, J_matrix: np.ndarray, h_field: np.ndarray, 
                 closures: Optional[List[int]] = None, 
                 tunneling_strength: float = 0.1,
                 entanglement_range: int = 3):
        """
        Initialize the Quantum Ising Traffic model.
        
        Args:
            num_nodes: Number of road segments
            J_matrix: Interaction matrix between road segments
            h_field: External field affecting each road segment
            closures: List of indices of permanently closed roads
            tunneling_strength: Strength of quantum tunneling effect
            entanglement_range: Range of entanglement-like correlations
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
        self.tunneling_strength = tunneling_strength
        self.entanglement_range = entanglement_range
        
        # Validate closures
        if not all(0 <= i < num_nodes for i in self.closures):
            raise ValueError("All closure indices must be between 0 and num_nodes-1")
        
        # Initialize quantum states (complex amplitudes)
        # Each road has amplitudes for |0⟩ and |1⟩ states
        self.amplitudes = np.zeros((self.N, 2), dtype=complex)
        
        # Initialize all roads to |1⟩ state (open) by default
        for i in range(self.N):
            if i in self.closures:
                self.amplitudes[i, 0] = 1.0  # |0⟩ state (closed)
                self.amplitudes[i, 1] = 0.0  # |1⟩ state (open)
            else:
                self.amplitudes[i, 0] = 0.0  # |0⟩ state (closed)
                self.amplitudes[i, 1] = 1.0  # |1⟩ state (open)
        
        # Initialize entanglement matrix
        self.entanglement_matrix = self._build_entanglement_matrix()
        
        # Initialize energy tracking
        self.energy_history = []
        self.current_energy = self.energy()
    
    def _build_entanglement_matrix(self) -> np.ndarray:
        """
        Build a matrix representing entanglement-like correlations between roads.
        Roads within entanglement_range of each other have non-zero correlations.
        """
        E = np.zeros((self.N, self.N), dtype=complex)
        
        # Create entanglement-like correlations based on distance
        for i in range(self.N):
            for j in range(i+1, self.N):
                # Calculate "distance" between roads (using J matrix as a proxy)
                if self.J[i, j] != 0:  # If roads are connected
                    # Add entanglement-like correlation
                    correlation = 1.0 / (1.0 + abs(i - j) / self.entanglement_range)
                    E[i, j] = correlation
                    E[j, i] = correlation  # Make symmetric
        
        return E
    
    def _get_state(self, i: int) -> int:
        """
        Measure the state of road i (collapse the wavefunction).
        Returns 0 (closed) or 1 (open).
        """
        if i in self.closures:
            return 0  # Closed roads are always measured as closed
        
        # Calculate probabilities
        p0 = abs(self.amplitudes[i, 0])**2
        p1 = abs(self.amplitudes[i, 1])**2
        
        # Normalize
        total = p0 + p1
        p0 /= total
        p1 /= total
        
        # Measure based on probabilities
        if random.random() < p0:
            # Collapse to |0⟩ state
            self.amplitudes[i, 0] = 1.0
            self.amplitudes[i, 1] = 0.0
            return 0
        else:
            # Collapse to |1⟩ state
            self.amplitudes[i, 0] = 0.0
            self.amplitudes[i, 1] = 1.0
            return 1
    
    def _get_all_states(self) -> np.ndarray:
        """
        Measure all road states (collapse all wavefunctions).
        Returns an array of 0s and 1s.
        """
        states = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            states[i] = self._get_state(i)
        return states
    
    def energy(self) -> float:
        """
        Calculate the expected energy of the quantum system.
        """
        # Get classical states for energy calculation
        states = self._get_all_states()
        
        # Create a mask for active roads
        active_mask = np.ones_like(states)
        active_mask[self.closures] = 0
        
        # Calculate interaction energy
        interaction = -0.5 * np.sum(
            self.J * np.outer(states * active_mask, states * active_mask)
        )
        
        # Calculate field energy
        field = -np.sum(self.h * states * active_mask)
        
        return interaction + field
    
    def _quantum_tunnel(self, i: int, T: float) -> bool:
        """
        Attempt quantum tunneling through energy barriers.
        Returns True if tunneling occurred.
        """
        if i in self.closures:
            return False  # Closed roads can't tunnel
        
        # Calculate energy barrier
        dE = self.delta_energy(i)
        
        # Quantum tunneling probability
        # Higher at low temperatures and for higher barriers
        tunneling_prob = self.tunneling_strength * np.exp(-dE / (T + 1e-10))
        
        return random.random() < tunneling_prob
    
    def delta_energy(self, i: int) -> float:
        """
        Calculate the energy change if road i is flipped.
        """
        if i in self.closures:
            return 0  # don't flip closed roads
        
        # Get current state
        current_state = self._get_state(i)
        
        # Calculate energy with current state
        E_current = -np.sum(self.J[i] * self._get_all_states()) - self.h[i]
        
        # Calculate energy with flipped state
        E_flipped = -np.sum(self.J[i] * self._get_all_states()) + self.h[i]
        
        return E_flipped - E_current
    
    def _apply_quantum_effects(self, i: int, T: float) -> None:
        """
        Apply quantum effects to road i:
        - Superposition
        - Entanglement
        - Interference
        """
        if i in self.closures:
            return  # Closed roads are unaffected
        
        # Get entangled roads
        entangled = np.where(self.entanglement_matrix[i] > 0)[0]
        
        # Create superposition
        phase = random.random() * 2 * np.pi
        self.amplitudes[i, 0] = cmath.exp(1j * phase) * np.sqrt(0.5)
        self.amplitudes[i, 1] = cmath.exp(1j * (phase + np.pi/2)) * np.sqrt(0.5)
        
        # Apply entanglement effects
        for j in entangled:
            if j != i and j not in self.closures:
                # Entangle amplitudes
                correlation = self.entanglement_matrix[i, j]
                self.amplitudes[j, 0] = correlation * self.amplitudes[i, 0]
                self.amplitudes[j, 1] = correlation * self.amplitudes[i, 1]
                
                # Normalize
                norm = np.sqrt(abs(self.amplitudes[j, 0])**2 + abs(self.amplitudes[j, 1])**2)
                self.amplitudes[j, 0] /= norm
                self.amplitudes[j, 1] /= norm
    
    def step(self, T: float) -> None:
        """
        Perform one quantum step of the simulation at temperature T.
        """
        i = random.randint(0, self.N - 1)
        
        # Apply quantum effects
        self._apply_quantum_effects(i, T)
        
        # Attempt quantum tunneling
        if self._quantum_tunnel(i, T):
            # Tunneling occurred, flip the state
            if self._get_state(i) == 0:
                self.amplitudes[i, 0] = 0.0
                self.amplitudes[i, 1] = 1.0
            else:
                self.amplitudes[i, 0] = 1.0
                self.amplitudes[i, 1] = 0.0
        else:
            # Classical thermal transition
            dE = self.delta_energy(i)
            if dE < 0 or random.random() < np.exp(-dE / T):
                # Flip the state
                if self._get_state(i) == 0:
                    self.amplitudes[i, 0] = 0.0
                    self.amplitudes[i, 1] = 1.0
                else:
                    self.amplitudes[i, 0] = 1.0
                    self.amplitudes[i, 1] = 0.0
        
        # Update energy
        self.current_energy = self.energy()
    
    def run(self, steps: int = 1000, T0: float = 5.0, cooling: float = 0.99, 
            track_energy: bool = True, track_interval: int = 100) -> np.ndarray:
        """
        Run the quantum simulation with specified parameters.
        
        Args:
            steps: Number of simulation steps
            T0: Initial temperature
            cooling: Temperature cooling rate
            track_energy: Whether to track energy history
            track_interval: How often to record energy (in steps)
            
        Returns:
            Final road states (0 = closed, 1 = open)
        """
        T = T0
        self.energy_history = []
        
        for step in range(steps):
            self.step(T)
            
            # Track energy if requested
            if track_energy and step % track_interval == 0:
                self.energy_history.append(self.current_energy)
            
            T *= cooling
        
        # Return final states
        return self._get_all_states()
    
    def get_energy_history(self) -> List[float]:
        """Return the history of energy values during the simulation."""
        return self.energy_history
    
    def get_quantum_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current quantum state of the system.
        Returns a dictionary with amplitudes and entanglement information.
        """
        return {
            'amplitudes': self.amplitudes.copy(),
            'entanglement': self.entanglement_matrix.copy(),
            'classical_states': self._get_all_states()
        }


if __name__ == "__main__":
    # Example usage
    N = 10  # number of roads
    J = np.random.rand(N, N) * 0.5
    J = (J + J.T) / 2  # make symmetric
    np.fill_diagonal(J, 0)

    h = np.random.randn(N) * 0.2  # some roads more attractive
    closures = [3, 7]  # pretend these roads are closed

    # Create quantum model
    model = QuantumIsingTraffic(
        num_nodes=N, 
        J_matrix=J, 
        h_field=h, 
        closures=closures,
        tunneling_strength=0.2,
        entanglement_range=2
    )
    
    # Run simulation
    final_state = model.run(track_energy=True)
    print("Final configuration:", final_state)
    print("Energy history:", model.get_energy_history())
    
    # Get quantum state
    quantum_state = model.get_quantum_state()
    print("Quantum amplitudes for first road:", quantum_state['amplitudes'][0]) 