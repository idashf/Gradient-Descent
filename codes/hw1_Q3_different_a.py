import math
import numpy as np
import matplotlib.pyplot as plt

# Define functions again
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
            break
    return m, b, cost_history

# Constants for step 3
K = 500
alpha = 1e-6
iterations = 50000
tolerance = 1e-4
a_values = [0.1, 0.3, 0.5]
colors = ['blue', 'orange', 'green']
markers = ['o', 's', '^']
linestyles = ['solid', 'dashed', 'dotted']

# Run experiments for different noise levels
results_noise = {}
for a in a_values:
    print(f"\nRunning experiment for a = {a}")
    x_train, y_train = data_generate(4, 8, K, a)
    m_final, b_final, cost_hist = gradient_descent(
        x_train, y_train, 0, 0, K, alpha, iterations, tolerance, compute_cost, compute_gradient
    )
    results_noise[f'a={a}'] = cost_hist

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Plot 1: all iterations (0–500)
for (label, cost_hist), color, linestyle, marker in zip(results_noise.items(), colors, linestyles, markers):
    ax1.plot(cost_hist[:500], label=label, linewidth=2.5, linestyle=linestyle, marker=marker,
             markevery=max(len(cost_hist[:500]) // 10, 1), markersize=6, color=color)

ax1.set_title('Full Convergence (First 500 Iterations)')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Cost')
ax1.set_yscale('log')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend(fontsize=9)

# Plot 2: zoomed between 100 and 300
for (label, cost_hist), color, linestyle, marker in zip(results_noise.items(), colors, linestyles, markers):
    ax2.plot(range(100, 301), cost_hist[100:301], label=label, linewidth=2.5, linestyle=linestyle, marker=marker,
            markevery=max(len(cost_hist[100:301]) // 10, 1), markersize=6, color=color)

ax2.set_title('Zoomed Convergence (Iterations 100–300)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_yscale('log')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend(fontsize=9)

plt.suptitle('Effect of Noise (a) on Gradient Descent Convergence', fontsize=16)
plt.savefig("step3_noise_comparison_combined.png", dpi=300)
plt.show()

