import numpy as np
import matplotlib.pyplot as plt

def diffusion_multiplier(rho, alpha, m_min):
    """
    Saturating diffusion reduction:
        m = 1 / (1 + alpha * rho)
        m = max(m, m_min)
    """
    m = 1.0 / (1.0 + alpha * rho)
    return np.maximum(m, m_min)


# Normalised fibre density rho = n_fibre / AVG_NETWORK_VOXEL_DENSITY
rho = np.linspace(0.0, 10.0, 400)

alphas = [0.25, 0.5, 1.0, 2.0]
m_mins = [0.01, 0.05, 0.1]

fig, axes = plt.subplots(1, len(m_mins), figsize=(15, 4), sharey=True)

for ax, m_min in zip(axes, m_mins):
    for alpha in alphas:
        m = diffusion_multiplier(rho, alpha, m_min)
        ax.plot(rho, m, label=f"alpha={alpha}")

    ax.set_title(f"m_min = {m_min}")
    ax.set_xlabel("rho = n_fibre / avg_density")
    ax.grid(True)

axes[0].set_ylabel("Diffusion multiplier m")
axes[-1].legend(title="alpha")

plt.tight_layout()
plt.show()
