import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict
from enum import Enum

class State(Enum):
    LOW_ENERGY_ABUNDANT_NO_RIVALS = 0
    MEDIUM_ENERGY_SCARCE_ONE_RIVAL = 1
    HIGH_ENERGY_ABUNDANT_MANY_RIVALS = 2

class Action(Enum):
    PASSIVE = 0
    TERRITORIAL = 1

class Animal:
    def __init__(
            self,
            learning_rate: float = 0.2,
            discount_factor: float = 0.9,
            epsilon: float = 0.2
    ):
        """Initialize an animal with Q-learning capabilities."""
        self.q_table = defaultdict(lambda: np.zeros(len(Action)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.energy = 50.0  # Start at middle energy level
        self.current_state = None
        self.last_action = None

    def choose_action(self, state: State) -> Action:
        """Choose an action using epsilon-greedy policy."""
        self.current_state = state

        if np.random.random() < self.epsilon:
            self.last_action = np.random.choice(list(Action))
        else:
            q_values = self.q_table[state]
            self.last_action = Action(np.argmax(q_values))

        return self.last_action

    def update_q_value(self, reward: float, next_state: State):
        """Update Q-value using TD learning."""
        if self.current_state is not None:
            current_q = self.q_table[self.current_state][self.last_action.value]
            next_max_q = np.max(self.q_table[next_state])

            # Q-learning update rule
            new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_max_q - current_q
            )

            self.q_table[self.current_state][self.last_action.value] = new_q

        self.current_state = next_state

    def update_energy(self, change: float):
        """Update animal's energy level with more dynamic range."""
        self.energy = max(0, min(200, self.energy + change))

class Environment:
    def __init__(
            self,
            resource_value: float = 20.0,
            territory_defense_cost: float = 5.0,
            contest_cost: float = 8.0,
            base_energy_cost: float = 2.0
    ):
        """Initialize the environment parameters."""
        self.resource_value = resource_value
        self.territory_defense_cost = territory_defense_cost
        self.contest_cost = contest_cost
        self.base_energy_cost = base_energy_cost

    def get_state(self, animal: Animal, nearby_animals: List[Animal]) -> State:
        """Determine the state of an animal based on its environment."""
        energy_level = animal.energy
        rival_count = len(nearby_animals)

        if energy_level < 70:  # Adjusted thresholds
            return State.LOW_ENERGY_ABUNDANT_NO_RIVALS
        elif energy_level < 140:
            return State.MEDIUM_ENERGY_SCARCE_ONE_RIVAL
        else:
            return State.HIGH_ENERGY_ABUNDANT_MANY_RIVALS

    def interact(self, animal1: Animal, animal2: Animal) -> Tuple[float, float]:
        """Handle interaction between two animals with modified payoffs."""
        action1 = animal1.last_action
        action2 = animal2.last_action

        # Base energy cost
        reward1 = -self.base_energy_cost
        reward2 = -self.base_energy_cost

        if action1 == Action.PASSIVE and action2 == Action.PASSIVE:
            # Share resource equally but with some inefficiency
            reward1 += self.resource_value * 0.4
            reward2 += self.resource_value * 0.4

        elif action1 == Action.TERRITORIAL and action2 == Action.PASSIVE:
            # Territorial gets more resource but pays defense cost
            reward1 += self.resource_value * 0.9 - self.territory_defense_cost
            reward2 += self.resource_value * 0.1

        elif action1 == Action.PASSIVE and action2 == Action.TERRITORIAL:
            # Territorial gets more resource but pays defense cost
            reward1 += self.resource_value * 0.1
            reward2 += self.resource_value * 0.9 - self.territory_defense_cost

        else:  # Both territorial
            # Contest for resource with higher stakes
            winner = np.random.choice([0, 1])
            if winner == 0:
                reward1 += self.resource_value - self.contest_cost
                reward2 += -self.contest_cost - self.territory_defense_cost
            else:
                reward1 += -self.contest_cost - self.territory_defense_cost
                reward2 += self.resource_value - self.contest_cost

        return reward1, reward2

class Simulation:
    def __init__(
            self,
            population_size: int,
            environment: Environment,
            interaction_rounds: int = 10
    ):
        """Initialize the simulation."""
        self.population = [Animal() for _ in range(population_size)]
        self.environment = environment
        self.interaction_rounds = interaction_rounds
        self.history = []

    def run_generation(self):
        """Run one generation of the simulation."""
        # Random pairs interact multiple times
        for _ in range(self.interaction_rounds):
            # Shuffle population for random pairing
            np.random.shuffle(self.population)

            # Pair animals and have them interact
            for i in range(0, len(self.population)-1, 2):
                animal1 = self.population[i]
                animal2 = self.population[i+1]

                # Get states
                state1 = self.environment.get_state(animal1, [animal2])
                state2 = self.environment.get_state(animal2, [animal1])

                # Choose actions
                animal1.choose_action(state1)
                animal2.choose_action(state2)

                # Get rewards from interaction
                reward1, reward2 = self.environment.interact(animal1, animal2)

                # Update energies
                animal1.update_energy(reward1)
                animal2.update_energy(reward2)

                # Update Q-values
                next_state1 = self.environment.get_state(animal1, [animal2])
                next_state2 = self.environment.get_state(animal2, [animal1])

                animal1.update_q_value(reward1, next_state1)
                animal2.update_q_value(reward2, next_state2)

        # Record current state
        self._record_state()

    def _record_state(self):
        """Record the current state of the population."""
        # Calculate strategy frequencies
        actions = [animal.last_action for animal in self.population if animal.last_action is not None]
        passive_freq = sum(1 for a in actions if a == Action.PASSIVE) / len(actions)
        territorial_freq = 1 - passive_freq

        # Calculate average energy
        avg_energy = np.mean([animal.energy for animal in self.population])

        self.history.append({
            'passive_freq': passive_freq,
            'territorial_freq': territorial_freq,
            'avg_energy': avg_energy
        })

    def plot_history(self):
        """Plot the evolution of strategies and average energy."""
        generations = range(len(self.history))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot strategy frequencies
        ax1.plot(generations, [h['passive_freq'] for h in self.history],
                 label='Passive')
        ax1.plot(generations, [h['territorial_freq'] for h in self.history],
                 label='Territorial')
        ax1.set_ylabel('Strategy Frequency')
        ax1.set_title('Evolution of Strategies')
        ax1.legend()
        ax1.grid(True)

        # Plot average energy
        ax2.plot(generations, [h['avg_energy'] for h in self.history],
                 label='Average Energy')
        ax2.set_ylabel('Energy Level')
        ax2.set_xlabel('Generation')
        ax2.set_title('Population Average Energy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

def run_simulation():
    """Run the simulation with the specified parameters."""
    # Create environment
    env = Environment(
        resource_value=20.0,
        territory_defense_cost=5.0,
        contest_cost=8.0,
        base_energy_cost=2.0
    )

    # Create simulation
    sim = Simulation(
        population_size=1000,
        environment=env,
        interaction_rounds=10
    )

    # Run simulation for 100 generations
    for _ in range(100):
        sim.run_generation()

    # Plot results
    sim.plot_history()

if __name__ == "__main__":
    run_simulation()