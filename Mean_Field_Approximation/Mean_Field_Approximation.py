import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class PublicGoodsGameMFA:
    def __init__(self, b, c, pc, p):
        """
        Initialize the Public Goods Game with Mean-Field Approximation

        Parameters:
        b  - benefit from public goods
        c  - cost of contributing
        pc - cost incurred by defectors when punished
        p  - cost of punishing
        """
        self.b = b  # benefit from public goods
        self.c = c  # cost of contributing
        self.pc = pc  # punishment cost for defectors
        self.p = p  # cost of punishing

    def payoffs(self, x):
        """
        Calculate payoffs for each strategy given population frequencies

        Parameters:
        x - array of [x_CP, x_DN, x_CN] frequencies
        """
        x_CP, x_DN, x_CN = x

        # Calculate payoffs for each strategy
        pi_CP = self.b * (x_CP + x_CN) - self.c - self.p * x_DN
        pi_DN = self.b * (x_CP + x_CN) - self.pc * x_CP
        pi_CN = self.b * (x_CP + x_CN) - self.c

        return np.array([pi_CP, pi_DN, pi_CN])

    def mean_payoff(self, x):
        """Calculate average payoff across the population"""
        individual_payoffs = self.payoffs(x)
        return np.sum(x * individual_payoffs)

    def replicator_dynamics(self, x, t):
        """
        Implement the replicator equation
        dx_i/dt = x_i * (π_i - π_bar)
        """
        pi = self.payoffs(x)
        pi_bar = self.mean_payoff(x)

        # Calculate derivatives
        dx = x * (pi - pi_bar)

        return dx

    def simulate(self, initial_frequencies, t_span, num_points=1000):
        """
        Simulate the evolution of strategy frequencies over time

        Parameters:
        initial_frequencies - initial distribution of strategies [x_CP, x_DN, x_CN]
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
        plt.plot(t, solution[:, 0], label='Cooperator-Punisher')
        plt.plot(t, solution[:, 1], label='Defector-Non-Punisher')
        plt.plot(t, solution[:, 2], label='Cooperator-Non-Punisher')
        plt.xlabel('Time')
        plt.ylabel('Strategy Frequency')
        plt.title('Evolution of Strategies in Public Goods Game')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
def run_simulation():
    # Set parameters
    b = 3.0   # benefit from public goods
    c = 1.0   # cost of contributing
    pc = 2.0  # punishment cost for defectors
    p = 0.5   # cost of punishing

    # Create game instance
    game = PublicGoodsGameMFA(b, c, pc, p)

    # Set initial frequencies and time span
    initial_frequencies = [0.33, 0.33, 0.34]  # Nearly equal initial distribution
    t_span = [0, 100]

    # Run simulation
    t, solution = game.simulate(initial_frequencies, t_span)

    # Plot results
    game.plot_evolution(t, solution)

    # Print final frequencies
    print("\nFinal strategy frequencies:")
    print(f"Cooperator-Punisher: {solution[-1, 0]:.3f}")
    print(f"Defector-Non-Punisher: {solution[-1, 1]:.3f}")
    print(f"Cooperator-Non-Punisher: {solution[-1, 2]:.3f}")

if __name__ == "__main__":
    run_simulation()