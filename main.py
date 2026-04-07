

# Clean, correct version of animated GBM simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm, gaussian_kde
from utils.gbm import simulate_gbm_paths

def animate_gbm():
    S0 = 100         # Initial price
    mu = 0.07        # Drift
    sigma = 0.2      # Volatility
    T = 1.0          # Time horizon (years)
    N = 252          # Time steps (daily)
    M = 200          # Number of paths
    random_seed = 42

    t, paths = simulate_gbm_paths(S0, mu, sigma, T, N, M, random_seed)
    expected = S0 * np.exp(mu * t)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 6))

    lines = [ax1.plot([], [], color='tab:blue', alpha=0.2, linewidth=1)[0] for _ in range(M)]
    expected_line, = ax1.plot([], [], color='red', lw=2, label='Expected Value $E[S_t]$')
    ax1.set_xlim(0, T)
    ax1.set_ylim(np.min(paths), np.max(paths)*1.05)
    ax1.set_title('Geometric Brownian Motion Paths (Animated)')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    def init():
        for line in lines:
            line.set_data([], [])
        expected_line.set_data([], [])
        return lines + [expected_line]

    def animate(frame):
        for i, line in enumerate(lines):
            line.set_data(t[:frame+1], paths[i, :frame+1])
        expected_line.set_data(t[:frame+1], expected[:frame+1])
        return lines + [expected_line]

    ani = FuncAnimation(fig, animate, frames=len(t), init_func=init, blit=True, interval=40, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_gbm()
