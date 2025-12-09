import math
import numpy as np
import matplotlib.pyplot as plt

# Core functions
def data_generate(m_main, b_main, K, a, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(0, 100, K)
    n = np.random.normal(0, a, K)
    y = m_main * x + b_main + n
    return x, y

def compute_cost(x, y, m, b, K):
    y_pred = m * x + b
    return np.mean((y_pred - y) ** 2) / 2

def compute_gradient(x, y, m, b):
    y_pred = m * x + b
    error = y_pred - y
    dl_dm = np.dot(error, x)
    dl_db = np.sum(error)
    return dl_dm, dl_db

def gradient_descent(x, y, m_in, b_in, K, alpha, num_iters, tol, cost_function, gradient_function):
    cost_history = []
    m = m_in
    b = b_in

    for i in range(int(num_iters)):
        dl_dm, dl_db = gradient_function(x, y, m, b)
        b -= alpha * dl_db
        m -= alpha * dl_dm
        cost = cost_function(x, y, m, b, K)
        cost_history.append(cost)
        if cost < tol:
            print(f"Early stopping at iteration {i} with cost {cost:.4e}")
            break

        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4}: Cost {cost:.2e}, m: {m:.3f}, b: {b:.3f}")

    return m, b, cost_history

# Configs for each K with tuned alpha and tolerance
experiment_configs = [
    {"K": 50, "alpha": 1e-5, "iterations": 50000, "tolerance": 1e-3},
    {"K": 500, "alpha": 1e-6, "iterations": 50000, "tolerance": 1e-4},
    {"K": 5000, "alpha": 1e-7, "iterations": 35000, "tolerance": 1e-4},
    {"K": 100000, "alpha": 5e-9, "iterations": 32000, "tolerance": 1e-3},
]

results = {}
for config in experiment_configs:
    K = config["K"]
    print(f"\nRunning experiment for K = {K}")
    x_train, y_train = data_generate(4, 8, K, a=0.1)
    m_final, b_final, cost_hist = gradient_descent(
        x_train, y_train, 0, 0, K,
        config["alpha"], config["iterations"], config["tolerance"],
        compute_cost, compute_gradient
    )
    label = f'K={K}, tol={config["tolerance"]}'
    results[label] = cost_hist

# Plot all K values with tolerance shown in label
colors = ['blue', 'orange', 'green', 'red']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
markers = ['o', 's', '^', 'D']

plt.figure(figsize=(10, 6))
for (label, cost_hist), color, linestyle, marker in zip(results.items(), colors, linestyles, markers):
    plt.plot(cost_hist[:500], label=label, linewidth=2, linestyle=linestyle, marker=marker,
             markevery=max(len(cost_hist[:500]) // 10, 1), markersize=5, color=color)

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.yscale('log')
plt.title('Cost over First 500 Iterations (Various K and Tolerance)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("step2_k_tolerance_comparison_log.png", dpi=300)
plt.show()

# Separate subplots
fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
for ax, ((label, cost_hist), color) in zip(axes, zip(results.items(), colors)):
    ax.plot(cost_hist[:500], color=color, linewidth=2)
    ax.set_title(label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.grid(True)

plt.suptitle("Cost Convergence for Different K and Tolerance (Subplots)", fontsize=16)
plt.savefig("step2_k_tolerance_comparison_subplots.png", dpi=300)
plt.show()
