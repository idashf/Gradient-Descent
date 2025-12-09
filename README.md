# Gradient Descent Analysis 

---

## 📌 Introduction
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

## 🚀 Step 1 — Sample Generation & Basic Gradient Descent
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

## 📊 Step 2 — Effect of Number of Samples (K) & Tolerance
To analyze the effect of dataset size, four values of \(K\) were tested:

