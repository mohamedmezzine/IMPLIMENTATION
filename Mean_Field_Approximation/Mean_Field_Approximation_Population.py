import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import comb

class PopulationPublicGoodsGameMFA:
    def __init__(self, b, c, group_size):
        """
        Initialize the Population Public Goods Game with Mean-Field Approximation

        Parameters:
        b - benefit multiplier for public goods (how much each contribution benefits everyone)
        c - cost of contributing
        group_size - number of players in each game group
        """
        self.b = b  # benefit multiplier
        self.c = c  # cost of contributing
        self.group_size = group_size

    def binomial_probability(self, k, n, p):
        """
        Calculate probability of k successes in n trials with probability p
        (i.e., probability of k cooperators in a group of n players)
        """
        return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    def expected_payoff(self, x_C):
        """
        Calculate expected payoffs for Cooperators and Defectors given cooperator frequency

        Parameters:
        x_C - frequency of Cooperators in the population

        Returns:
        Tuple of (cooperator_payoff, defector_payoff)
        """
        n = self.group_size - 1  # number of other players in group

        # Initialize payoffs
        payoff_C = -self.c  # cooperator starts with paying cost
        payoff_D = 0        # defector starts with no cost

        # Calculate expected benefit based on binomial distribution of other cooperators
        for k in range(n + 1):
            # Probability of k cooperators among other n players
            prob = self.binomial_probability(k, n, x_C)

            # For cooperator: add benefit from k other cooperators plus own contribution
            benefit_C = self.b * (k + 1) / self.group_size
            payoff_C += prob * benefit_C

            # For defector: add benefit from k cooperators
            benefit_D = self.b * k / self.group_size
            payoff_D += prob * benefit_D

        return np.array([payoff_C, payoff_D])

    def replicator_dynamics(self, x, t):
        """
        Implement the replicator equation for population dynamics
        dx_i/dt = x_i * (π_i - π_bar)
        """
        x_C = x[0]  # frequency of cooperators
        x_D = x[1]  # frequency of defectors

        # Calculate payoffs
        payoffs = self.expected_payoff(x_C)

        # Calculate average payoff
        avg_payoff = x_C * payoffs[0] + x_D * payoffs[1]

        # Calculate derivatives
        dx_C = x_C * (payoffs[0] - avg_payoff)
        dx_D = x_D * (payoffs[1] - avg_payoff)

        return np.array([dx_C, dx_D])

    def simulate(self, initial_frequencies, t_span, num_points=1000):
        """
        Simulate the evolution of strategy frequencies over time

        Parameters:
        initial_frequencies - initial distribution of strategies [x_C, x_D]
        t_span - time span for simulation [t_start, t_end]
        num_points - number of time points to simulate
        """
        t = np.linspace(t_span[0], t_span[1], num_points)

        # Ensure initial frequencies sum to 1
        initial_frequencies = np.array(initial_frequencies)
        initial_frequencies = initial_frequencies / np.sum(initial_frequencies)

        # Solve the system of differential equations
        solution = odeint(self.replicator_dynamics, initial_frequencies, t)

        return t, solution

    def plot_evolution(self, t, solution):
        """Plot the evolution of strategy frequencies over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(t, solution[:, 0], 'g-', label='Cooperators')
        plt.plot(t, solution[:, 1], 'r-', label='Defectors')
        plt.xlabel('Time')
        plt.ylabel('Strategy Frequency')
        plt.title(f'Evolution of Strategies in Population Public Goods Game\n(Group Size: {self.group_size})')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_payoff_landscape(self):
        """Plot the payoff landscape as a function of cooperator frequency"""
        x_C = np.linspace(0, 1, 100)
        payoffs_C = []
        payoffs_D = []

        for freq in x_C:
            payoffs = self.expected_payoff(freq)
            payoffs_C.append(payoffs[0])
            payoffs_D.append(payoffs[1])

        plt.figure(figsize=(10, 6))
        plt.plot(x_C, payoffs_C, 'g-', label='Cooperator Payoff')
        plt.plot(x_C, payoffs_D, 'r-', label='Defector Payoff')
        plt.xlabel('Frequency of Cooperators')
        plt.ylabel('Expected Payoff')
        plt.title(f'Payoff Landscape\n(Group Size: {self.group_size})')
        plt.legend()
        plt.grid(True)
        plt.show()

def run_simulation():
    # Set parameters
    b = 3.0   # benefit multiplier
    c = 1.0   # cost of contributing
    group_size = 5  # number of players in each group

    # Create game instance
    game = PopulationPublicGoodsGameMFA(b, c, group_size)

    # Set initial frequencies and time span
    initial_frequencies = [0.7, 0.3]  # Start with more cooperators
    t_span = [0, 50]

    # Run simulation
    t, solution = game.simulate(initial_frequencies, t_span)

    # Plot results
    game.plot_evolution(t, solution)
    game.plot_payoff_landscape()

    # Print final frequencies
    print("\nFinal strategy frequencies:")
    print(f"Cooperators: {solution[-1, 0]:.3f}")
    print(f"Defectors: {solution[-1, 1]:.3f}")

    # Print game parameters
    print("\nGame parameters:")
    print(f"Benefit multiplier (b): {b}")
    print(f"Cost of cooperation (c): {c}")
    print(f"Group size: {group_size}")

if __name__ == "__main__":
    run_simulation()