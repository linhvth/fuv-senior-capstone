

This document is dedicated to record my thinking flow of experiement design for my Capstone project on SGD.

## 1, Motivation
### 1.1, Why do we need to conduct experiments?

### 1.2. What are our goals? And what need to be done to achieve those goals?
- To compare the results produced by our implemented algorithm with the theoretical results.
- To measure effects of different parameters on the algorithm. It can be seen as to test the flexibility/robustness of the algorithm.

### 1.3. Experiments
#### a. Assumptions

This experiments is built based on the following assumptions:
- In this current version, we assume the gradient of function $f$ is L-Lipschitz in L2 norm (Euclidean distance) and there exist a $\theta^*$ such that $f(\theta^*)$ is minimized. Without this assumptions, this experiment is invalid due to the complexity of the problem.
- Actually, for simplicity, we only consider L2 norm.
- Unbiased Gradient, that means for any stochastic vectors produced by any methods implemented in this experiment, those stochastic vectors are unbiased, i.e. the expectation of a stochastic vector at a location `t-1` is equal to the true gradient of the function taken at this location.

## 2. Methodology

## 3. Results

## 4. Discussion
