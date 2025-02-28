import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.signal import savgol_filter
import statsmodels.api as sm
from typing import List, Tuple, Optional
import seaborn as sns

class GameVisualizer:
    def __init__(self, hawk_freqs: List[float], dove_freqs: List[float], payoffs: List[float]):
        """Initialize visualizer with simulation results."""
        self.hawk_freqs = np.array(hawk_freqs)
        self.dove_freqs = np.array(dove_freqs)
        self.payoffs = np.array(payoffs)
        self.time_steps = np.arange(len(hawk_freqs))

    def moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average with specified window size."""
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_smoothed_trends(self, window_size: int = 50):
        """Plot smoothed trends with confidence intervals."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Calculate smoothed data
        smooth_hawk = self.moving_average(self.hawk_freqs, window_size)
        smooth_dove = self.moving_average(self.dove_freqs, window_size)
        smooth_payoff = self.moving_average(self.payoffs, window_size)

        # Calculate rolling standard deviation
        valid_length = len(smooth_hawk)
        x_smooth = np.arange(valid_length)

        # Plot frequencies
        ax1.plot(self.time_steps, self.hawk_freqs, 'r-', alpha=0.2, label='Raw Hawk')
        ax1.plot(self.time_steps, self.dove_freqs, 'b-', alpha=0.2, label='Raw Dove')
        ax1.plot(x_smooth, smooth_hawk, 'r-', linewidth=2, label='Smoothed Hawk')
        ax1.plot(x_smooth, smooth_dove, 'b-', linewidth=2, label='Smoothed Dove')

        ax1.set_title('Strategy Frequencies with Moving Average')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Plot payoffs
        ax2.plot(self.time_steps, self.payoffs, 'g-', alpha=0.2, label='Raw Payoff')
        ax2.plot(x_smooth, smooth_payoff, 'g-', linewidth=2, label='Smoothed Payoff')
        ax2.set_title('Average Payoff with Moving Average')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Payoff')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_time_series_decomposition(self, period: int = 50):
        """Plot time series decomposition for hawk frequency."""
        # Create pandas Series
        hawk_series = pd.Series(self.hawk_freqs)

        # Perform decomposition
        decomposition = sm.tsa.seasonal_decompose(hawk_series, period=period)

        # Plot components
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))

        ax1.plot(self.time_steps, hawk_series, label='Original', color='red')
        ax1.set_title('Original Hawk Frequency')
        ax1.legend()

        ax2.plot(self.time_steps, decomposition.trend, label='Trend', color='blue')
        ax2.set_title('Trend Component')
        ax2.legend()

        ax3.plot(self.time_steps, decomposition.seasonal, label='Seasonal', color='green')
        ax3.set_title('Seasonal Component')
        ax3.legend()

        ax4.plot(self.time_steps, decomposition.resid, label='Residual', color='purple')
        ax4.set_title('Residual Component')
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def plot_state_space(self):
        """Create state-space plot of hawk vs dove frequencies."""
        plt.figure(figsize=(10, 10))

        # Plot state space trajectory
        plt.plot(self.hawk_freqs, self.dove_freqs, 'k-', alpha=0.5, linewidth=0.7)

        # Add points for start and end
        plt.plot(self.hawk_freqs[0], self.dove_freqs[0], 'go', label='Start', markersize=10)
        plt.plot(self.hawk_freqs[-1], self.dove_freqs[-1], 'ro', label='End', markersize=10)

        # Add direction arrows periodically
        step = len(self.hawk_freqs) // 20  # Show 20 arrows along the trajectory
        for i in range(0, len(self.hawk_freqs)-step, step):
            plt.arrow(self.hawk_freqs[i], self.dove_freqs[i],
                      self.hawk_freqs[i+step]-self.hawk_freqs[i],
                      self.dove_freqs[i+step]-self.dove_freqs[i],
                      head_width=0.02, head_length=0.02, fc='k', ec='k', alpha=0.5)

        plt.xlabel('Hawk Frequency')
        plt.ylabel('Dove Frequency')
        plt.title('State-Space Plot of Strategy Frequencies')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.show()

    def plot_phase_portrait_with_payoff(self):
        """Create 3D phase portrait including payoff."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D trajectory
        ax.plot3D(self.hawk_freqs, self.dove_freqs, self.payoffs, 'k-', alpha=0.5)

        # Add points for start and end
        ax.scatter([self.hawk_freqs[0]], [self.dove_freqs[0]], [self.payoffs[0]],
                   color='g', s=100, label='Start')
        ax.scatter([self.hawk_freqs[-1]], [self.dove_freqs[-1]], [self.payoffs[-1]],
                   color='r', s=100, label='End')

        ax.set_xlabel('Hawk Frequency')
        ax.set_ylabel('Dove Frequency')
        ax.set_zlabel('Average Payoff')
        ax.set_title('3D Phase Portrait with Payoff')
        ax.legend()
        plt.show()

    def plot_all_visualizations(self):
        """Generate all visualizations in sequence."""
        print("Generating smoothed trends plot...")
        self.plot_smoothed_trends()

        print("Generating time series decomposition...")
        self.plot_time_series_decomposition()

        print("Generating state space plot...")
        self.plot_state_space()

        print("Generating 3D phase portrait...")
        self.plot_phase_portrait_with_payoff()

# Example usage:
if __name__ == "__main__":
    # Simulate some data (replace with your actual simulation results)
    from Experience_Weighted_Attraction import Population, GameParams, EWAParams

    # Run simulation
    game_params = GameParams(V=2.0, C=3.0)
    ewa_params = EWAParams(phi=0.95, delta=0.5, rho=0.95, lambda_=2.0)
    pop = Population(n_agents=100, game_params=game_params, ewa_params=ewa_params)
    history, payoff_history = pop.run_simulation(n_steps=1000)

    # Extract frequencies
    hawk_freqs, dove_freqs = zip(*history)

    # Create visualizer and generate all plots
    visualizer = GameVisualizer(hawk_freqs, dove_freqs, payoff_history)
    visualizer.plot_all_visualizations()