import math
import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic linear data with Gaussian noise
def data_generate(m_main, b_main, K, a, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(0, 100, K)
    n = np.random.normal(0, a, K)
    y = m_main * x + b_main + n
    return x, y

# vectorized cost function
def compute_cost(x, y, m, b, K):
    y_pred = m * x + b
    return np.mean((y_pred - y) ** 2) / 2

# Function to compute gradients of the cost function with respect to m and b
def compute_gradient(x, y, m, b):
    y_pred = m * x + b
    error = y_pred - y
    dl_dm = np.dot(error, x)
    dl_db = np.sum(error)
    return dl_dm, dl_db

# Function to perform gradient descent and optimize m and b
def gradient_descent(x, y, m_in, b_in, K, alpha, num_iters, cost_function, gradient_function):
    cost_history = []
    params_history = []

    m = m_in
    b = b_in

    for i in range(int(num_iters)):
        dl_dm, dl_db = gradient_function(x, y, m, b)
        b -= alpha * dl_db
        m -= alpha * dl_dm
        cost_history.append(cost_function(x, y, m, b, K))
        params_history.append([m, b])

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {cost_history[-1]:0.2e} ",
                  f"dl_dm: {dl_dm: 0.3e}, dl_db: {dl_db: 0.3e}  ",
                  f"m: {m: 0.3e}, b:{b: 0.5e}")

    return m, b, cost_history, params_history

# Parameters
m_main = 4
b_main = 8
K = 50
a = 0.1
iterations = 50000
learning_rate = 1e-5
m_start = 0
b_start = 0

# Generate training data
x_train, y_train = data_generate(m_main, b_main, K, a)

# Run gradient descent optimization
m_final, b_final, cost_hist, params_hist = gradient_descent(
    x_train, y_train, m_start, b_start, K, learning_rate, iterations, compute_cost, compute_gradient
)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(18, 5))

# Plot 1: Cost at start
ax1.plot(cost_hist[:100])
ax1.set_title("Cost vs. iteration (start)")
ax1.set_ylabel('Cost')
ax1.set_xlabel('Iteration step')
ax1.grid(True)

# Plot 2: Cost at end
if len(cost_hist) > 1000:
    ax2.plot(np.arange(1000, len(cost_hist)), cost_hist[1000:])
else:
    ax2.plot(np.arange(len(cost_hist)), cost_hist)
ax2.set_title("Cost vs. iteration (end)")
ax2.set_ylabel('Cost')
ax2.set_xlabel('Iteration step')
ax2.grid(True)

# Plot 3: Data and model
x_line = np.linspace(0, 100, 100)
y_true_line = m_main * x_line + b_main
y_estimated_line = m_final * x_line + b_final

ax3.scatter(x_train, y_train, label='Noisy samples', color='blue', alpha=0.6)
ax3.plot(x_line, y_true_line, label='True line (m=4, b=8)', color='green', linestyle='--')
ax3.plot(x_line, y_estimated_line, label='Estimated line', color='red', linewidth=2)
ax3.set_title('Data & Model Comparison')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.legend()
ax3.grid(True)

plt.suptitle("Gradient Descent - STEP 1 ", fontsize=16)
plt.show()
