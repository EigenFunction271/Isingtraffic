import numpy as np
import random
from typing import List, Optional, Tuple, Dict
import cmath
from scipy.linalg import expm

class QuantumIsingTraffic:
    """
    A quantum-inspired version of the Ising Traffic model.
    
    This implementation extends the classical Ising model with quantum-inspired effects:
    1. Superposition states: Roads can exist in a superposition of open/closed states
    2. Quantum tunneling: Roads can "tunnel" through energy barriers
    3. Entanglement-like correlations: Roads can be correlated beyond direct connections
    4. Quantum interference: Paths can interfere constructively or destructively
    
    Mathematical foundation:
    - State vector: |ψ⟩ = α|0⟩ + β|1⟩ where |α|² + |β|² = 1
    - Energy: E = -∑ᵢⱼ Jᵢⱼsᵢsⱼ - ∑ᵢ hᵢsᵢ + quantum_terms
    - Time evolution: |ψ(t)⟩ = exp(-iHt/ℏ)|ψ(0)⟩
    """
    
    def __init__(self, num_nodes: int, J_matrix: np.ndarray, h_field: np.ndarray, 
                 closures: Optional[List[int]] = None, 
                 tunneling_strength: float = 0.2,
                 entanglement_range: int = 3):
        """
        Initialize the quantum-inspired Ising Traffic model.
        
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
        
        # Initialize quantum state vectors (complex128 to handle complex amplitudes)
        self.amplitudes = np.zeros((self.N, 2), dtype=np.complex128)
        
        # Initialize spins (classical representation)
        self.spins = np.ones(self.N)
        for i in self.closures:
            self.spins[i] = -1  # permanently closed roads
        
        # Initialize all roads in a superposition state
        for i in range(self.N):
            if i in self.closures:
                # Closed roads are fixed in the closed state
                self.amplitudes[i, 0] = 1.0
                self.amplitudes[i, 1] = 0.0
            else:
                # Open roads start in a superposition state
                self.amplitudes[i, 0] = 1.0 / np.sqrt(2)
                self.amplitudes[i, 1] = 1.0 / np.sqrt(2)
        
        # Initialize energy tracking
        self.energy_history = []
        self.current_energy = self.energy()
        self.entanglement_history = []
    
    def get_quantum_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current quantum state of the system.
        
        Returns:
        - amplitudes: Matrix of quantum amplitudes [α, β] for each road
        - phases: Matrix of quantum phases [θ₀, θ₁] for each road
        """
        return {
            'amplitudes': self.amplitudes.copy(),
            'phases': np.angle(self.amplitudes)
        }
    
    def get_quantumness(self, i: int) -> float:
        """
        Calculate the "quantumness" of a road (how quantum-like its state is).
        
        Mathematical definition:
        quantumness = 1 - |⟨ψ|σᵢ|ψ⟩|²
        where σᵢ is the Pauli-Z operator for road i
        
        This measures how far the state is from a classical state (0 or 1).
        """
        # |⟨ψ|σᵢ|ψ⟩|² = |α|² - |β|²
        classical_component = abs(np.abs(self.amplitudes[i, 0])**2 - np.abs(self.amplitudes[i, 1])**2)
        return 1 - classical_component
    
    def get_current_spins(self) -> np.ndarray:
        """
        Get the current classical representation of the road states.
        
        Returns:
        - spins: Array of road states (1 for open, -1 for closed)
        """
        return self.spins.copy()
    
    def get_average_entanglement(self) -> float:
        """
        Calculate the average entanglement between roads.
        
        Mathematical definition:
        E = (1/N)∑ᵢⱼ |⟨ψ|σᵢσⱼ|ψ⟩ - ⟨ψ|σᵢ|ψ⟩⟨ψ|σⱼ|ψ⟩|
        """
        if len(self.entanglement_history) > 0:
            return np.mean(self.entanglement_history[-1])
        return 0.0
    
    def energy(self) -> float:
        """
        Calculate the total energy of the system including quantum terms.
        
        Mathematical formula:
        E = -∑ᵢⱼ Jᵢⱼ⟨ψ|σᵢσⱼ|ψ⟩ - ∑ᵢ hᵢ⟨ψ|σᵢ|ψ⟩ + quantum_corrections
        
        Quantum corrections include:
        1. Tunneling energy: -γ∑ᵢ ⟨ψ|σₓ|ψ⟩
        2. Entanglement energy: -Jₑ∑ᵢⱼₖ ⟨ψ|σᵢσⱼσₖ|ψ⟩
        """
        # Classical energy terms
        active_mask = np.ones_like(self.spins)
        active_mask[self.closures] = 0
        
        # Calculate expectation values
        exp_values = np.array([np.abs(a[1])**2 - np.abs(a[0])**2 for a in self.amplitudes])
        
        # Interaction energy
        interaction = -0.5 * np.sum(
            self.J * np.outer(exp_values * active_mask, exp_values * active_mask)
        )
        
        # Field energy
        field = -np.sum(self.h * exp_values * active_mask)
        
        # Quantum tunneling energy
        tunneling = -self.tunneling_strength * np.sum(
            [2 * np.real(self.amplitudes[i, 0] * np.conj(self.amplitudes[i, 1]))
             for i in range(self.N) if i not in self.closures]
        )
        
        # Entanglement-like energy
        entanglement = 0
        for i in range(self.N):
            if i in self.closures:
                continue
            for j in range(max(0, i-self.entanglement_range), 
                         min(self.N, i+self.entanglement_range+1)):
                if j == i or j in self.closures:
                    continue
                entanglement += np.abs(
                    exp_values[i] * exp_values[j] - 
                    np.mean(exp_values[max(0, i-self.entanglement_range):
                                     min(self.N, i+self.entanglement_range+1)])
                )
        
        return interaction + field + tunneling + 0.1 * entanglement
    
    def delta_energy(self, i: int) -> float:
        """
        Calculate the energy change if road i's state is modified.
        
        Mathematical formula:
        ΔE = 2sᵢ(∑ⱼ Jᵢⱼsⱼ + hᵢ) + quantum_terms
        
        Quantum terms include:
        1. Tunneling contribution: -γ(⟨ψ'|σₓ|ψ'⟩ - ⟨ψ|σₓ|ψ⟩)
        2. Entanglement contribution: -Jₑ∑ⱼₖ(⟨ψ'|σᵢσⱼσₖ|ψ'⟩ - ⟨ψ|σᵢσⱼσₖ|ψ⟩)
        """
        if i in self.closures:
            return 0
        
        # Classical energy change
        exp_value = np.abs(self.amplitudes[i, 1])**2 - np.abs(self.amplitudes[i, 0])**2
        dE = 2 * exp_value * (np.sum(self.J[i] * self.spins) + self.h[i])
        
        # Quantum tunneling contribution
        dE += self.tunneling_strength * (
            2 * np.real(self.amplitudes[i, 0] * np.conj(self.amplitudes[i, 1]))
        )
        
        return dE
    
    def step(self, T: float) -> None:
        """
        Perform one step of the quantum-inspired simulation at temperature T.
        Closed roads are never selected for flipping.
        """
        i = random.randint(0, self.N - 1)
        if i in self.closures:
            return  # Skip closed roads
        
        # Calculate energy change if we were to flip this road
        dE = self.delta_energy(i)
        
        # Quantum tunneling probability
        tunneling_prob = np.exp(-dE / T) * self.tunneling_strength
        
        # Apply quantum tunneling effect
        if random.random() < tunneling_prob:
            # Phase rotation based on energy change
            phase = -dE / T
            # Ensure we're working with complex128
            self.amplitudes[i] = self.amplitudes[i].astype(np.complex128) * np.exp(1j * phase)
            
            # Normalize amplitudes
            norm = np.sqrt(np.sum(np.abs(self.amplitudes[i])**2))
            self.amplitudes[i] /= norm
            
            # Update classical spin based on measurement
            if np.abs(self.amplitudes[i, 1])**2 > np.abs(self.amplitudes[i, 0])**2:
                self.spins[i] = 1  # Open
            else:
                self.spins[i] = -1  # Closed
        
        # Apply entanglement-like correlations
        self._apply_entanglement(i)
        
        # Update current energy
        self.current_energy = self.energy()
    
    def _apply_entanglement(self, i: int) -> None:
        """
        Apply entanglement-like correlations to nearby roads.
        
        This method is called after a road's state changes to propagate
        the change to nearby roads within the entanglement range.
        """
        if i in self.closures:
            return
        
        # Get the range of roads to affect
        start = max(0, i - self.entanglement_range)
        end = min(self.N, i + self.entanglement_range + 1)
        
        # Calculate correlation strength based on distance
        for j in range(start, end):
            if j == i or j in self.closures:
                continue
            
            # Correlation strength decreases with distance
            distance = abs(j - i)
            correlation = 1.0 / (1.0 + distance)
            
            # Apply correlation to amplitudes
            self.amplitudes[j] = (1 - correlation) * self.amplitudes[j] + correlation * self.amplitudes[i]
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(self.amplitudes[j])**2))
            self.amplitudes[j] /= norm
            
            # Update classical spin based on measurement
            if np.abs(self.amplitudes[j, 1])**2 > np.abs(self.amplitudes[j, 0])**2:
                self.spins[j] = 1  # Open
            else:
                self.spins[j] = -1  # Closed
    
    def run(self, steps: int = 1000, T0: float = 5.0, cooling: float = 0.99,
            track_energy: bool = True, track_interval: int = 100) -> np.ndarray:
        """
        Run the quantum-inspired simulation.
        
        Mathematical process:
        1. Initialize temperature T = T₀
        2. For each step:
           - Apply quantum step
           - Cool temperature: T' = T * cooling
           - Track energy and entanglement
        3. Return final classical configuration
        """
        T = T0
        self.energy_history = []
        self.entanglement_history = []
        
        for step in range(steps):
            self.step(T)
            
            if track_energy and step % track_interval == 0:
                self.energy_history.append(self.current_energy)
                # Calculate current entanglement
                entanglement = []
                for i in range(self.N):
                    if i in self.closures:
                        continue
                    for j in range(max(0, i-self.entanglement_range),
                                 min(self.N, i+self.entanglement_range+1)):
                        if j == i or j in self.closures:
                            continue
                        entanglement.append(
                            abs(np.abs(self.amplitudes[i, 1])**2 - 
                                np.abs(self.amplitudes[j, 1])**2)
                        )
                self.entanglement_history.append(np.mean(entanglement))
            
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