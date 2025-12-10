# Gradient Descent Analysis 

---

##  Introduction
The goal of this assignment is to implement and analyze the **Gradient Descent** algorithm for estimating the parameters of a linear model using randomly generated data.  
Four experimental steps were designed to investigate the effect of several parameters on the learning process, including:

- Number of samples (K)  
- Noise level (a)  
- Learning rate (α)  
- Tolerance  
- Number of iterations  

The implementation was done in **Python**, using the following libraries:

- **NumPy** – numerical computations, gradient calculation, random data generation  
- **Matplotlib** – plotting and visualization  
- **Math** – basic numerical formatting  

---

##  Step 1 — Sample Generation & Basic Gradient Descent
Random data was generated based on a linear model:

\[
y = mx + b + \text{noise}
\]

with true parameters:

- \( m = 8 \)  
- \( b = 4 \)

Noise was drawn from a normal distribution with mean 0 and standard deviation 0.1.

Gradient Descent was implemented to estimate these parameters.  
The algorithm includes:

- Defining the MSE cost function  
- Computing gradients w.r.t. \(m\) and \(b\)  
- Updating parameters iteratively  

The model successfully converged to the true parameters using an appropriate learning rate and sufficient iterations.  
Plots were generated showing:

- Cost decrease at the beginning and end of training  
- Comparison between the true line and the estimated line  

---

##  Step 2 — Effect of Number of Samples (K) & Tolerance
To analyze the effect of dataset size, four values of \(K\) were tested:
K = 50, 500, 5000, 100000

For each K, learning rate, tolerance, and number of iterations were tuned individually to avoid divergence and ensure smooth convergence.

### Key findings:
- Increasing **K** leads to more **stable convergence**.  
- Larger K results in **smaller final error**.  
- However, larger datasets require a **smaller learning rate** and more computation time.  

A logarithmic plot of cost during the first 500 iterations was used to compare all values of K.

---

##  Step 3 — Effect of Noise Level (a)
This step evaluated the sensitivity of Gradient Descent to noise in the data.  
Three noise values were tested:
a = 0.1, 0.3, 0.5

with \(K = 500\).

### Observations:
- Higher noise → higher initial cost  
- Convergence becomes slower and less stable  
- Zoomed-in plots (iterations 100–300) show this effect more clearly  

---

##  Step 4 — Final Interpretation
The assignment concluded with the question:

> **Which parameter has the greatest impact on accurately estimating the model parameters?**

Options included:

- Number of samples  
- Number of iterations  
- Tolerance  
- Learning rate  

###  Correct answer: **Learning Rate**

A learning rate that is too large → divergence  
A learning rate that is too small → extremely slow learning  

Thus, selecting a proper learning rate plays the most critical role in the success of Gradient Descent.

---

##  Conclusion
In this assignment:

- Gradient Descent was fully implemented from scratch  
- The influence of dataset size, noise, tolerance, and learning rate was systematically studied  
- Experiments showed that although many factors affect learning performance, the **learning rate** has the most significant impact on achieving stable and accurate convergence  

This assignment offered hands-on experience and deeper intuition about optimization algorithms in machine learning.

---




