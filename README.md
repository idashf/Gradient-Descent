# Gradient-Descent
Introduction
The goal of this assignment is to implement and analyze the Gradient Descent algorithm for estimating the parameters of a linear model using randomly generated samples. Four different experimental steps were designed in order to investigate the effect of various parameters such as the number of samples (K), noise level (a), learning rate, acceptable error (tolerance), and number of iterations on the learning process.
This assignment was implemented in Python using the numpy and matplotlib libraries.

Used Libraries

Numpy: for numerical computations, generating random data, and calculating gradients.

Matplotlib.pyplot: for plotting curves to better understand the learning behavior.

Math: for basic calculations and formatting numerical outputs.
Step 1: Sample Generation and Basic Gradient Descent Implementation

In this step, random data was generated using the linear relation:

𝑦
=
𝑚
𝑥
+
𝑏
+
noise
y=mx+b+noise

The true values of the parameters were set to:

𝑚
=
8
m=8

𝑏
=
4
b=4

Noise was drawn from a normal distribution with mean 0 and standard deviation 0.1.
Then, the Gradient Descent algorithm was implemented to estimate these two parameters.
The pipeline includes defining the MSE cost function, computing its gradients with respect to 
𝑚
m and 
𝑏
b, and updating the parameters iteratively using those gradients.

The implementation successfully recovered the true parameters using a suitable learning rate 
(
𝛼
)
(α) and enough iterations.
Additionally, the decrease of cost at the beginning and the end of training, as well as the comparison between the true line and the predicted line, were plotted.

Step 2: Effect of Number of Samples (K) and Tolerance

In this step, the goal was to investigate how the number of training samples affects the accuracy and convergence speed of Gradient Descent.
Four values of 
𝐾
K were examined:

𝐾
=
50
,
 
500
,
 
5000
,
 
100000
K=50, 500, 5000, 100000

For each value of 
𝐾
K, the learning rate, tolerance, and number of iterations were carefully tuned to avoid divergence or extremely slow convergence.

The cost variation during training for each 
𝐾
K was plotted.
The experiments showed:

Increasing 
𝐾
K results in more stable convergence and lower final error.

However, larger 
𝐾
K requires a smaller learning rate and more training time.

A logarithmic plot of the first 500 iterations was presented for a clearer comparison.

Step 3: Effect of Noise Level (a)

The purpose of this step was to examine how the added noise affects the behavior of Gradient Descent.
Three noise levels were tested:

𝑎
=
0.1
,
 
0.3
,
 
0.5
a=0.1, 0.3, 0.5

The number of samples was fixed at 
𝐾
=
500
K=500 and other hyperparameters were kept constant.

Observations:

As noise increased, the initial cost became larger.

The algorithm’s convergence became slower and less stable.

Two plots (full range and zoomed 100–300 iterations) were used to clearly show this effect.

Step 4: Final Interpretation

At the end of the assignment, the question was:
Which factor has the greatest impact on accurately estimating the model parameters?

The available options were:

Number of samples

Number of iterations

Appropriate learning rate

Tolerance

Based on the experiments, the correct answer is:

✅ The learning rate is the most critical factor.

If the learning rate is too large → divergence
If the learning rate is too small → extremely slow convergence

Therefore, choosing the right learning rate plays the most essential role in the success of the algorithm.

Conclusion

In this assignment, the Gradient Descent algorithm was fully implemented and analyzed.
By using randomly generated data and controlling parameters such as learning rate, sample size, noise level, and tolerance, the performance of the algorithm was thoroughly evaluated.

It was concluded that although all parameters influence the learning process, the learning rate has the most crucial impact on achieving correct and efficient convergence.
This assignment provided practical experience for deeper understanding of foundational optimization algorithms.
