import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class Individual:
    def __init__(self, p_contribute: float = 0.5):
        """Initialize an individual with probability of contributing."""
        self.p_contribute = p_contribute
        self.payoff = 0.0
        self.last_action = None

    def choose_action(self) -> bool:
        """Choose whether to contribute based on current probability."""
        self.last_action = np.random.random() < self.p_contribute
        return self.last_action

    def update_strategy(self, observed_proportion: float, learning_rate: float):
        """Update contribution probability using proportional imitation."""
        self.p_contribute += learning_rate * (observed_proportion - self.p_contribute)
        # Ensure probability stays between 0 and 1
        self.p_contribute = np.clip(self.p_contribute, 0, 1)

class PublicGoodsGame:
    def __init__(
            self,
            population_size: int,
            group_size: int,
            network_type: str = "small-world",
            contribution_cost: float = 1.0,
            multiplication_factor: float = 2.0,
            learning_rate: float = 0.1,
            neighbor_count: int = 10
    ):
        """Initialize the Public Goods Game simulation."""
        self.population_size = population_size
        self.group_size = group_size
        self.contribution_cost = contribution_cost
        self.multiplication_factor = multiplication_factor
        self.learning_rate = learning_rate
        self.neighbor_count = neighbor_count

        # Initialize population
        self.population = [Individual() for _ in range(population_size)]

        # Create social network
        if network_type == "small-world":
            self.network = nx.watts_strogatz_graph(population_size, neighbor_count, 0.1)
        elif network_type == "random":
            self.network = nx.erdos_renyi_graph(population_size, neighbor_count/population_size)
        else:  # regular lattice
            self.network = nx.grid_2d_graph(int(np.sqrt(population_size)), int(np.sqrt(population_size)))

        self.history = []

    def play_round(self):
        """Play one round of the public goods game."""
        # Reset payoffs
        for individual in self.population:
            individual.payoff = 0

        # Create random groups
        indices = np.random.permutation(self.population_size)
        groups = [indices[i:i + self.group_size] for i in range(0, len(indices), self.group_size)]

        # Play public goods game in each group
        for group in groups:
            self._play_group_game(group)

        # Update strategies based on neighborhood observations
        self._update_strategies()

        # Record current state
        self._record_state()

    def _play_group_game(self, group_indices: List[int]):
        """Play public goods game within a group."""
        # Get contributions
        contributions = [self.population[i].choose_action() for i in group_indices]
        total_contributions = sum(contributions) * self.contribution_cost

        # Calculate group benefit
        benefit_per_person = (total_contributions * self.multiplication_factor) / len(group_indices)

        # Assign payoffs
        for idx, contributed in zip(group_indices, contributions):
            individual = self.population[idx]
            individual.payoff = benefit_per_person - (contributed * self.contribution_cost)

    def _update_strategies(self):
        """Update strategies based on neighborhood observations."""
        for idx in range(self.population_size):
            # Get neighbors
            neighbors = list(self.network.neighbors(idx))
            if not neighbors:
                continue

            # Calculate average payoff in neighborhood
            neighbor_payoffs = [self.population[n].payoff for n in neighbors]
            avg_payoff = np.mean(neighbor_payoffs)

            # Count successful contributors and free-riders among neighbors
            successful_contributors = sum(
                1 for n in neighbors
                if self.population[n].last_action and self.population[n].payoff > avg_payoff
            )
            successful_total = sum(
                1 for n in neighbors
                if self.population[n].payoff > avg_payoff
            )

            # Calculate proportion of successful contributors
            if successful_total > 0:
                proportion_contribute = successful_contributors / successful_total
                self.population[idx].update_strategy(proportion_contribute, self.learning_rate)

    def _record_state(self):
        """Record current state of the population."""
        avg_p_contribute = np.mean([ind.p_contribute for ind in self.population])
        actual_contributions = np.mean([1 if ind.last_action else 0 for ind in self.population])
        avg_payoff = np.mean([ind.payoff for ind in self.population])

        self.history.append({
            'avg_p_contribute': avg_p_contribute,
            'actual_contributions': actual_contributions,
            'avg_payoff': avg_payoff
        })

    def plot_history(self):
        """Plot the evolution of cooperation and payoffs."""
        generations = range(len(self.history))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot cooperation probabilities and actual contributions
        ax1.plot(generations, [h['avg_p_contribute'] for h in self.history],
                 label='Average P(Contribute)')
        ax1.plot(generations, [h['actual_contributions'] for h in self.history],
                 label='Actual Contributions')
        ax1.set_ylabel('Cooperation Rate')
        ax1.set_title('Evolution of Cooperation')
        ax1.legend()
        ax1.grid(True)

        # Plot average payoffs
        ax2.plot(generations, [h['avg_payoff'] for h in self.history],
                 label='Average Payoff')
        ax2.set_ylabel('Payoff')
        ax2.set_xlabel('Generation')
        ax2.set_title('Evolution of Payoffs')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# Example usage
def run_simulation():
    # Create simulation
    game = PublicGoodsGame(
        population_size=100,
        group_size=5,
        network_type="small-world",
        contribution_cost=1.0,
        multiplication_factor=3.0,
        learning_rate=0.1,
        neighbor_count=10
    )

    # Run simulation for 100 generations
    for _ in range(100):
        game.play_round()

    # Plot results
    game.plot_history()

if __name__ == "__main__":
    run_simulation()