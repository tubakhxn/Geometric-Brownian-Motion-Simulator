import numpy as np

def simulate_gbm_paths(S0, mu, sigma, T, N, M, random_seed=None):
    """
    Simulate M paths of Geometric Brownian Motion.
    
    Parameters:
        S0 (float): Initial stock price
        mu (float): Drift coefficient
        sigma (float): Volatility coefficient
        T (float): Time horizon (in years)
        N (int): Number of time steps
        M (int): Number of paths
        random_seed (int, optional): Seed for reproducibility
    Returns:
        t (np.ndarray): Time points (shape: [N+1])
        paths (np.ndarray): Simulated paths (shape: [M, N+1])
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    dt = T / N
    t = np.linspace(0, T, N+1)
    # Brownian increments
    dW = np.random.normal(0, np.sqrt(dt), size=(M, N))
    W = np.cumsum(dW, axis=1)
    W = np.hstack([np.zeros((M, 1)), W])
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * W
    paths = S0 * np.exp(drift + diffusion)
    return t, paths
