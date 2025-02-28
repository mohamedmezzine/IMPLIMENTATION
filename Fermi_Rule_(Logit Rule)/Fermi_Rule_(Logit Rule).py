import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, strategies: List[str], initial_q_value: float = 0.0):
        """Initialize an agent with Q-values for each strategy."""
        self.q_values = {strategy: initial_q_value for strategy in strategies}

    def select_strategy(self, temperature: float) -> str:
        """Select a strategy using Boltzmann exploration."""
        # Calculate probabilities using Boltzmann distribution
        values = np.array(list(self.q_values.values()))
        probabilities = np.exp(values / temperature)
        probabilities = probabilities / np.sum(probabilities)

        # Select strategy based on probabilities
        strategies = list(self.q_values.keys())
        return np.random.choice(strategies, p=probabilities)

    def update_q_value(self, strategy: str, reward: float, learning_rate: float):
        """Update Q-value for the chosen strategy."""
        current_q = self.q_values[strategy]
        self.q_values[strategy] = current_q + learning_rate * (reward - current_q)

class BoltzmannEGT:
    def __init__(
            self,
            population_size: int,
            strategies: List[str],
            payoff_matrix: Dict[Tuple[str, str], Tuple[float, float]],
            initial_temperature: float = 1.0,
            learning_rate: float = 0.1,
            temperature_decay: float = 0.999
    ):
        """Initialize the evolutionary game simulation."""
        self.population = [Agent(strategies) for _ in range(population_size)]
        self.strategies = strategies
        self.payoff_matrix = payoff_matrix
        self.temperature = initial_temperature
        self.learning_rate = learning_rate
        self.temperature_decay = temperature_decay
        self.history = []

    def run_generation(self) -> Dict[str, float]:
        """Run one generation of the simulation."""
        # Randomly pair agents
        indices = np.random.permutation(len(self.population))
        pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1, 2)]

        # For each pair, play the game
        for i1, i2 in pairs:
            agent1, agent2 = self.population[i1], self.population[i2]

            # Select strategies using Boltzmann exploration
            strategy1 = agent1.select_strategy(self.temperature)
            strategy2 = agent2.select_strategy(self.temperature)

            # Get payoffs from payoff matrix
            payoff1, payoff2 = self.payoff_matrix[(strategy1, strategy2)]

            # Update Q-values
            agent1.update_q_value(strategy1, payoff1, self.learning_rate)
            agent2.update_q_value(strategy2, payoff2, self.learning_rate)

        # Calculate strategy distribution
        strategy_counts = self._get_strategy_distribution()

        # Record current state
        self.history.append(strategy_counts)

        # Decay temperature
        self.temperature *= self.temperature_decay

        return strategy_counts

    def _get_strategy_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of strategies in the population."""
        strategy_counts = {strategy: 0 for strategy in self.strategies}

        for agent in self.population:
            strategy = max(agent.q_values.items(), key=lambda x: x[1])[0]
            strategy_counts[strategy] += 1

        # Convert to percentages
        total = len(self.population)
        return {k: v/total for k, v in strategy_counts.items()}

    def plot_history(self):
        """Plot the evolution of strategy distributions over time."""
        generations = range(len(self.history))

        plt.figure(figsize=(10, 6))
        for strategy in self.strategies:
            strategy_history = [gen[strategy] for gen in self.history]
            plt.plot(generations, strategy_history, label=strategy)

        plt.xlabel('Generation')
        plt.ylabel('Strategy Frequency')
        plt.title('Evolution of Strategies Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
def run_simulation():
    # Define parameters
    population_size = 100
    strategies = ['Cooperate', 'Defect']
    payoff_matrix = {
        ('Cooperate', 'Cooperate'): (3, 3),
        ('Cooperate', 'Defect'): (0, 5),
        ('Defect', 'Cooperate'): (5, 0),
        ('Defect', 'Defect'): (1, 1)
    }

    # Create simulation
    simulation = BoltzmannEGT(
        population_size=population_size,
        strategies=strategies,
        payoff_matrix=payoff_matrix,
        initial_temperature=2.0,
        learning_rate=0.1,
        temperature_decay=0.995
    )

    # Run simulation for 100 generations
    for _ in range(100):
        strategy_dist = simulation.run_generation()

    # Plot results
    simulation.plot_history()

if __name__ == "__main__":
    run_simulation()