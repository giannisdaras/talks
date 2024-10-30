import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("whitegrid")

def brownian_motion(total_time, num_steps):
    time_steps = torch.linspace(0, total_time, num_steps)
    delta_t = time_steps[1] - time_steps[0]
    increments = torch.randn(num_steps - 1, 2) * torch.sqrt(delta_t)
    increments = torch.cat([torch.zeros(1, 2), increments], dim=0)
    return time_steps, torch.cumsum(increments, dim=0)

def wiener_process(tt):
    """Generate a Wiener process trajectory."""
    dt = np.diff(tt, prepend=tt[0])
    dW = np.random.normal(0, np.sqrt(dt))
    return np.cumsum(dW)

def brownian_bridge(num_points, x0, xT, T):
    """Generate a Brownian bridge trajectory."""
    tt = np.linspace(0, T, num_points)
    W = wiener_process(tt)
    process = x0 + W + (tt/T) * (xT - x0 - W[-1])
    return tt, process


if __name__ == "__main__":
    tt, B = brownian_bridge(num_points=100, x0=0, xT=0, T=1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(tt, B, 'b-', alpha=1.0, color='orange')
    plt.axis('off')  # Turn off all axes
    plt.savefig("brownian_bridge.svg", transparent=True, bbox_inches='tight')
