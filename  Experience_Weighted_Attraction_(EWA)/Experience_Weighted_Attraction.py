import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
    phi: float    # Forgetting rate
    delta: float  # Weight on foregone payoffs
    rho: float    # Experience weight
    lambda_: float  # Choice sensitivity

class Agent:
    def __init__(self, ewa_params: EWAParams):
        self.params = ewa_params
        # Initialize attractions with small random values to break symmetry
        self.attractions = np.random.normal(0, 0.1, 2)
        self.experience = 1.0
        self.last_payoff = 0.0  # Track last received payoff

    def choose_action(self) -> int:
        """Choose action based on softmax of attractions."""
        probs = self._get_choice_probabilities()
        return np.random.choice([0, 1], p=probs)

    def _get_choice_probabilities(self) -> np.ndarray:
        """Calculate probabilities using softmax."""
        # Add numerical stability
        max_attr = np.max(self.attractions)
        exp_attractions = np.exp(self.params.lambda_ * (self.attractions - max_attr))
        return exp_attractions / np.sum(exp_attractions)

    def update(self, action: int, opponent_action: int, payoff_matrix: np.ndarray):
        """Update attractions based on EWA learning rule."""
        # Calculate actual and foregone payoffs
        actual_payoff = payoff_matrix[action, opponent_action]
        foregone_action = 1 - action
        foregone_payoff = payoff_matrix[foregone_action, opponent_action]

        # Store the actual payoff received
        self.last_payoff = actual_payoff

        # Update experience counter
        new_experience = self.params.rho * self.experience + 1

        # Calculate new attractions
        for s in [0, 1]:
            # Determine if this was the chosen action
            is_chosen = (s == action)

            # Calculate reinforcement
            if is_chosen:
                reinforcement = actual_payoff
            else:
                reinforcement = foregone_payoff

            # Apply EWA update rule
            self.attractions[s] = (
                    (self.params.phi * self.experience * self.attractions[s] +
                     (self.params.delta + (1 - self.params.delta) * is_chosen) * reinforcement)
                    / new_experience
            )

        # Update experience
        self.experience = new_experience

class Population:
    def __init__(
            self,
            n_agents: int,
            game_params: GameParams,
            ewa_params: EWAParams
    ):
        self.agents = [Agent(ewa_params) for _ in range(n_agents)]
        self.game_params = game_params
        self.payoff_matrix = game_params.get_payoff_matrix()
        self.history = []
        self.payoff_history = []
        self.n_agents = n_agents

    def step(self) -> Tuple[float, float, float]:
        """Run one step of the simulation."""
        # Get all agents' actions
        actions = np.array([agent.choose_action() for agent in self.agents])

        # Create random pairs
        indices = np.random.permutation(self.n_agents)
        if len(indices) % 2 == 1:  # If odd number of agents, one sits out
            indices = indices[:-1]

        pairs = indices.reshape(-1, 2)

        # Update each pair
        for i, j in pairs:
            action_i = actions[i]
            action_j = actions[j]

            # Update both agents
            self.agents[i].update(action_i, action_j, self.payoff_matrix)
            self.agents[j].update(action_j, action_i, self.payoff_matrix)

        # Calculate frequencies
        hawk_freq = np.mean(actions == 0)
        dove_freq = 1 - hawk_freq

        # Calculate average population payoff
        avg_payoff = np.mean([agent.last_payoff for agent in self.agents])

        self.history.append((hawk_freq, dove_freq))
        self.payoff_history.append(avg_payoff)

        return hawk_freq, dove_freq, avg_payoff

    def run_simulation(self, n_steps: int) -> Tuple[List[Tuple[float, float]], List[float]]:
        """Run simulation for specified number of steps."""
        for _ in range(n_steps):
            self.step()
        return self.history, self.payoff_history

    def plot_results(self):
        """Plot the evolution of strategy frequencies and average payoff."""
        hawk_freqs, dove_freqs = zip(*self.history)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot strategy frequencies
        ax1.plot(hawk_freqs, label='Hawk', color='red')
        ax1.plot(dove_freqs, label='Dove', color='blue')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Strategy Frequency')
        ax1.set_title('Evolution of Strategy Frequencies')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(0, 1)

        # Plot average payoff
        ax2.plot(self.payoff_history, label='Average Payoff', color='green')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Average Payoff')
        ax2.set_title('Population Average Payoff')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()


def calculate_rolling_average(self, data, window=50):
    """Calculate rolling average with the specified window."""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_results(self):
    """Plot the evolution of strategy frequencies and average payoff with improved visualization."""
    hawk_freqs, dove_freqs = zip(*self.history)

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Raw data plot (left column)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # Smoothed data plot (right column)
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    # Window size for rolling average
    window = 50

    # Plot raw data (left side)
    ax1.plot(hawk_freqs, 'r-', alpha=0.3, label='Raw Hawk')
    ax1.plot(dove_freqs, 'b-', alpha=0.3, label='Raw Dove')
    ax1.set_title('Raw Strategy Frequencies')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)

    ax2.plot(self.payoff_history, 'g-', alpha=0.3, label='Raw Payoff')
    ax2.set_title('Raw Average Payoff')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Payoff')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Calculate and plot smoothed data (right side)
    smooth_hawk = self.calculate_rolling_average(hawk_freqs, window)
    smooth_dove = self.calculate_rolling_average(dove_freqs, window)
    smooth_payoff = self.calculate_rolling_average(self.payoff_history, window)

    # Add confidence intervals
    hawk_std = np.array([np.std(hawk_freqs[max(0, i-window):i+1])
                         for i in range(window-1, len(hawk_freqs))])
    dove_std = np.array([np.std(dove_freqs[max(0, i-window):i+1])
                         for i in range(window-1, len(dove_freqs))])
    payoff_std = np.array([np.std(self.payoff_history[max(0, i-window):i+1])
                           for i in range(window-1, len(self.payoff_history))])

    x_smooth = np.arange(len(smooth_hawk))

    # Plot smoothed frequencies with confidence intervals
    ax3.plot(x_smooth, smooth_hawk, 'r-', label='Smoothed Hawk', linewidth=2)
    ax3.plot(x_smooth, smooth_dove, 'b-', label='Smoothed Dove', linewidth=2)
    ax3.fill_between(x_smooth, smooth_hawk - hawk_std, smooth_hawk + hawk_std,
                     color='r', alpha=0.2)
    ax3.fill_between(x_smooth, smooth_dove - dove_std, smooth_dove + dove_std,
                     color='b', alpha=0.2)
    ax3.set_title(f'Smoothed Strategy Frequencies (Window={window})')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1)

    # Plot smoothed payoff with confidence interval
    ax4.plot(x_smooth, smooth_payoff, 'g-', label='Smoothed Payoff', linewidth=2)
    ax4.fill_between(x_smooth, smooth_payoff - payoff_std, smooth_payoff + payoff_std,
                     color='g', alpha=0.2)
    ax4.set_title(f'Smoothed Average Payoff (Window={window})')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Payoff')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add statistics to the plot
    stats_text = (
        f'Final 100 steps statistics:\n'
        f'Hawk: {np.mean(hawk_freqs[-100:]):.3f} ± {np.std(hawk_freqs[-100:]):.3f}\n'
        f'Dove: {np.mean(dove_freqs[-100:]):.3f} ± {np.std(dove_freqs[-100:]):.3f}\n'
        f'Payoff: {np.mean(self.payoff_history[-100:]):.3f} ± {np.std(self.payoff_history[-100:]):.3f}'
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace')

    plt.tight_layout()
    plt.show()

    # Print additional statistics
    print("\nFinal Population Statistics (last 100 steps):")
    print(f"Hawk frequency: {np.mean(hawk_freqs[-100:]):.3f} ± {np.std(hawk_freqs[-100:]):.3f}")
    print(f"Dove frequency: {np.mean(dove_freqs[-100:]):.3f} ± {np.std(dove_freqs[-100:]):.3f}")
    print(f"Average payoff: {np.mean(self.payoff_history[-100:]):.3f} ± {np.std(self.payoff_history[-100:]):.3f}")

# Example usage remains the same
if __name__ == "__main__":
    game_params = GameParams(V=2.0, C=3.0)
    ewa_params = EWAParams(phi=0.95, delta=0.5, rho=0.95, lambda_=2.0)

    pop = Population(n_agents=100, game_params=game_params, ewa_params=ewa_params)
    pop.run_simulation(n_steps=1000)
    pop.plot_results()