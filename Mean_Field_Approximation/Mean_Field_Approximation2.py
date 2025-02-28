import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class BasicPublicGoodsGameMFA:
    def __init__(self, b, c):
        """
        Initialize the Basic Public Goods Game with Mean-Field Approximation

        Parameters:
        b - benefit multiplier for public goods (how much each contribution benefits everyone)
        c - cost of contributing
        """
        self.b = b  # benefit multiplier
        self.c = c  # cost of contributing

    def payoffs(self, x):
        """
        Calculate payoffs for Cooperators and Defectors given population frequencies

        Parameters:
        x - array of [x_C, x_D] frequencies where:
            x_C: frequency of Cooperators
            x_D: frequency of Defectors

        Returns:
        Array of [payoff_C, payoff_D]
        """
        x_C = x[0]  # frequency of cooperators

        # Cooperators get benefit from all cooperators (including themselves) but pay cost
        payoff_C = self.b * x_C - self.c

        # Defectors get benefit from cooperators but don't pay cost
        payoff_D = self.b * x_C

        return np.array([payoff_C, payoff_D])

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
        plt.title('Evolution of Strategies in Basic Public Goods Game')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_phase_space(self, num_points=20):
        """Plot the phase space of the system"""
        # Create grid of points
        x_C = np.linspace(0, 1, num_points)
        X_C, X_D = np.meshgrid(x_C, 1 - x_C)

        # Calculate derivatives at each point
        dX = np.zeros((num_points, num_points, 2))
        for i in range(num_points):
            for j in range(num_points):
                x = np.array([X_C[i,j], X_D[i,j]])
                dX[i,j] = self.replicator_dynamics(x, 0)

        plt.figure(figsize=(8, 8))
        plt.quiver(X_C, X_D, dX[:,:,0], dX[:,:,1])
        plt.xlabel('Frequency of Cooperators')
        plt.ylabel('Frequency of Defectors')
        plt.title('Phase Space of Basic Public Goods Game')
        plt.grid(True)
        plt.show()

def run_simulation():
    # Set parameters
    b = 1.5  # benefit multiplier
    c = 1.0  # cost of contributing

    # Create game instance
    game = BasicPublicGoodsGameMFA(b, c)

    # Set initial frequencies and time span
    initial_frequencies = [0.7, 0.3]  # Start with more cooperators
    t_span = [0, 50]

    # Run simulation
    t, solution = game.simulate(initial_frequencies, t_span)

    # Plot results
    game.plot_evolution(t, solution)

    # Plot phase space
    game.plot_phase_space()

    # Print final frequencies
    print("\nFinal strategy frequencies:")
    print(f"Cooperators: {solution[-1, 0]:.3f}")
    print(f"Defectors: {solution[-1, 1]:.3f}")

    # Print game parameters
    print("\nGame parameters:")
    print(f"Benefit multiplier (b): {b}")
    print(f"Cost of cooperation (c): {c}")

if __name__ == "__main__":
    run_simulation()