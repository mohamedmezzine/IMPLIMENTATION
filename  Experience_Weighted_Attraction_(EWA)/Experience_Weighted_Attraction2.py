import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

@dataclass
class GameParams:
    V: float  # Value of resource
    C: float  # Cost of fighting

    def get_payoff_matrix(self) -> np.ndarray:
        """Returns the payoff matrix for Hawk-Dove game."""
        return np.array([
            [(self.V - self.C)/2, self.V],  # Hawk payoffs
            [0, self.V/2]                    # Dove payoffs
        ])

@dataclass
class EWAParams:
    phi: float    # Experience decay (memory) parameter [0,1]
    delta: float  # Weight on foregone payoffs [0,1]
    rho: float    # Growth rate of experience [0,1]
    lambda_: float  # Payoff sensitivity (exploration-exploitation) > 0
    initial_attraction: float = 0.0  # Initial attraction value

class Agent:
    def __init__(self, ewa_params: EWAParams):
        self.params = ewa_params
        # Initialize attractions with the same value to avoid initial bias
        self.attractions = np.full(2, self.params.initial_attraction)
        self.experience = 1.0
        self.last_payoff = 0.0

    def choose_action(self, temperature: float = 1.0) -> int:
        """Choose action using softmax with temperature scaling."""
        probs = self._get_choice_probabilities(temperature)
        return np.random.choice([0, 1], p=probs)

    def _get_choice_probabilities(self, temperature: float) -> np.ndarray:
        """Calculate choice probabilities using softmax with temperature."""
        scaled_attractions = self.params.lambda_ * self.attractions / temperature
        max_attr = np.max(scaled_attractions)
        exp_attractions = np.exp(scaled_attractions - max_attr)
        return exp_attractions / np.sum(exp_attractions)

    def update(self, action: int, opponent_action: int, payoff_matrix: np.ndarray):
        """Update attractions using EWA learning rule with improved numerical stability."""
        # Calculate actual and foregone payoffs
        actual_payoff = payoff_matrix[action, opponent_action]
        foregone_payoff = payoff_matrix[1-action, opponent_action]

        # Store actual payoff
        self.last_payoff = actual_payoff

        # Update experience
        new_experience = self.params.rho * self.experience + 1

        # Calculate decay factor
        decay = self.params.phi * (self.experience / new_experience)

        # Update attractions with improved numerical stability
        for s in [0, 1]:
            is_chosen = (s == action)
            reinforcement = actual_payoff if is_chosen else foregone_payoff

            # Calculate learning weight
            learning_weight = (is_chosen + (1 - is_chosen) * self.params.delta) / new_experience

            # Update attraction
            self.attractions[s] = (decay * self.attractions[s] + learning_weight * reinforcement)

        # Update experience counter
        self.experience = new_experience

class Population:
    def __init__(
            self,
            n_agents: int,
            game_params: GameParams,
            ewa_params: EWAParams,
            temperature_schedule: callable = lambda t: 1.0
    ):
        self.agents = [Agent(ewa_params) for _ in range(n_agents)]
        self.game_params = game_params
        self.payoff_matrix = game_params.get_payoff_matrix()
        self.history = []
        self.payoff_history = []
        self.n_agents = n_agents
        self.temperature_schedule = temperature_schedule
        self.current_step = 0

    def step(self) -> Tuple[float, float, float]:
        """Run one step of the simulation with temperature scheduling."""
        self.current_step += 1
        temperature = self.temperature_schedule(self.current_step)

        # Get all agents' actions
        actions = np.array([agent.choose_action(temperature) for agent in self.agents])

        # Create random pairs for interaction
        indices = np.random.permutation(self.n_agents)
        if len(indices) % 2 == 1:
            indices = indices[:-1]
        pairs = indices.reshape(-1, 2)

        # Update each pair
        for i, j in pairs:
            action_i, action_j = actions[i], actions[j]
            self.agents[i].update(action_i, action_j, self.payoff_matrix)
            self.agents[j].update(action_j, action_i, self.payoff_matrix)

        # Calculate statistics
        hawk_freq = np.mean(actions == 0)
        dove_freq = 1 - hawk_freq
        avg_payoff = np.mean([agent.last_payoff for agent in self.agents])

        self.history.append((hawk_freq, dove_freq))
        self.payoff_history.append(avg_payoff)

        return hawk_freq, dove_freq, avg_payoff

    def run_simulation(self, n_steps: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Run simulation for specified number of steps."""
        for _ in range(n_steps):
            self.step()
        return self.history, self.payoff_history

def plot_results(history: List[Tuple[float, float]], payoff_history: List[float]):
    """Plot the evolution of strategy frequencies and average payoff with improved visualization."""
    hawk_freqs, dove_freqs = zip(*history)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate moving averages for smoother plots
    window = 50
    hawk_ma = np.convolve(hawk_freqs, np.ones(window)/window, mode='valid')
    dove_ma = np.convolve(dove_freqs, np.ones(window)/window, mode='valid')
    payoff_ma = np.convolve(payoff_history, np.ones(window)/window, mode='valid')

    # Plot strategy frequencies
    time_steps = np.arange(len(hawk_freqs))
    ma_steps = np.arange(len(hawk_ma))

    # Raw data with low alpha
    ax1.plot(time_steps, hawk_freqs, 'r-', alpha=0.3, label='Hawk (raw)')
    ax1.plot(time_steps, dove_freqs, 'b-', alpha=0.3, label='Dove (raw)')

    # Moving averages with high alpha
    ax1.plot(ma_steps + window//2, hawk_ma, 'r-', linewidth=2, label='Hawk (MA)')
    ax1.plot(ma_steps + window//2, dove_ma, 'b-', linewidth=2, label='Dove (MA)')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Strategy Frequency')
    ax1.set_title('Evolution of Strategy Frequencies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Plot average payoff
    ax2.plot(time_steps, payoff_history, 'g-', alpha=0.3, label='Raw')
    ax2.plot(ma_steps + window//2, payoff_ma, 'g-', linewidth=2, label='Moving Average')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Average Payoff')
    ax2.set_title('Population Average Payoff')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add final statistics
    final_stats = (
        f'Final Statistics (last 100 steps):\n'
        f'Hawk: {np.mean(hawk_freqs[-100:]):.3f} ± {np.std(hawk_freqs[-100:]):.3f}\n'
        f'Dove: {np.mean(dove_freqs[-100:]):.3f} ± {np.std(dove_freqs[-100:]):.3f}\n'
        f'Payoff: {np.mean(payoff_history[-100:]):.3f} ± {np.std(payoff_history[-100:]):.3f}'
    )
    plt.figtext(0.02, 0.02, final_stats, fontsize=10, family='monospace')

    plt.tight_layout()
    plt.show()

# Example usage
def main():
    # Define game parameters for Hawk-Dove game
    game_params = GameParams(
        V=2.0,  # Value of resource
        C=3.0   # Cost of fighting (higher than value)
    )

    # Define EWA parameters with more stable learning
    ewa_params = EWAParams(
        phi=0.9,     # High memory retention
        delta=0.5,   # Equal weight to actual and foregone payoffs
        rho=0.95,    # Slower experience growth
        lambda_=5.0, # Higher payoff sensitivity
        initial_attraction=1.0  # Non-zero initial attraction
    )

    # Define temperature schedule for better exploration-exploitation balance
    def temperature_schedule(t):
        return max(1.0, 10.0 * np.exp(-0.001 * t))

    # Create population with more agents for better statistics
    pop = Population(
        n_agents=200,
        game_params=game_params,
        ewa_params=ewa_params,
        temperature_schedule=temperature_schedule
    )

    # Run simulation for more steps
    history, payoff_history = pop.run_simulation(n_steps=2000)

    # Plot results
    plot_results(history, payoff_history)

if __name__ == "__main__":
    main()