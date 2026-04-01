# Julia Applied Machine Learning

A collection of supervised, unsupervised, and reinforcement learning projects built in Julia, covering classical machine learning algorithms implemented from scratch alongside library-based approaches. Each subfolder is a standalone project targeting a different algorithm or technique.

---

## Repository Structure
```
julia-applied-machine-learning/
├── Linear_Regression_Model/
│   ├── non_ml_approach.jl
│   ├── ml_approach.jl
│   └── housing_data.csv
├── Logistic_Regression_Model/
│   ├── logistic_curve.jl
│   ├── logistic_regression.jl
│   └── wolfspider.csv
├── Naive_Bayes_Classifier/
│   ├── data_wrangling.jl
│   ├── naive_bayes.jl
│   └── tennis.csv
├── Support_Vector_Machine/
│   └── svm_iris.jl
├── Decision_Tree_Models/
│   ├── decision_tree.jl
│   └── advanced_tree_models.jl
├── K_Nearest_Neighbors/
│   ├── knn_demo.jl
│   └── knn_iris.jl
├── Neural_Networks/
│   └── mnist_mlp.jl
├── Clustering/
│   ├── kmeans_demo.jl
│   └── kmeans_cats.jl
├── Principal_Component_Analysis/
│   ├── pca_demo.jl
│   └── pca_iris.jl
├── Markov_Decision_Processes/
│   └── grid_world_mdp.jl
└── Reinforcement_Learning/
    ├── sarsa.jl
    └── qlearning.jl
```

---

## Projects

### Linear Regression
Two approaches to the same problem — predicting housing prices in Portland from square footage. `non_ml_approach.jl` fits a model analytically using GLM.jl in a single closed-form solve. `ml_approach.jl` implements batch gradient descent from scratch with feature normalization, training over 1000 epochs, and plots both the regression line and the cost curve. Both models are evaluated side by side using R² and RMSE.

---

### Logistic Regression
Binary classification of wolf spider habitat presence based on sand grain size. `logistic_curve.jl` builds intuition for the sigmoid function and its parameterization. `logistic_regression.jl` implements batch gradient descent with binary cross-entropy loss, training the decision boundary iteratively and visualizing the evolving fit alongside the cost curve.

---

### Naive Bayes Classifier
Classifies whether tennis should be played given weather conditions (outlook, temperature, humidity, wind). `data_wrangling.jl` loads and processes the dataset using CSV and DataFrames. `naive_bayes.jl` implements the full classifier from scratch using Bayes' theorem with Laplace smoothing, organized as a generalized function that accepts any combination of feature inputs.

---

### Support Vector Machine
Multi-class classification on the Iris dataset using LIBSVM.jl. Trains SVM models with Linear, RBF, and Polynomial kernels and compares their test accuracy side by side. Includes an 80/20 stratified train/test split and a confusion matrix per model.

---

### Decision Tree Models
`decision_tree.jl` trains a depth-2 decision tree classifier on Iris using DecisionTree.jl, with per-class stratified sampling, tree structure printing, and prediction confidence scores. `advanced_tree_models.jl` extends this to Random Forest and AdaBoost, comparing both models head to head with a side-by-side prediction breakdown and class probability outputs.

---

### K-Nearest Neighbors
`knn_demo.jl` visualizes the k-NN concept on random 2D data — building a k-d tree, finding the k nearest neighbors to a test point, and drawing spoke lines from the test point to each neighbor. `knn_iris.jl` applies k-NN classification to the Iris dataset using majority vote over k=5 neighbors with accuracy evaluation.

---

### Neural Networks
Trains a multilayer perceptron on the MNIST handwritten digit dataset using Flux.jl. Architecture is 784 → 32 (ReLU) → 10 (softmax) trained with ADAM and cross-entropy loss over 500 epochs. Outputs test accuracy, a per-prediction breakdown, and a learning curve plot.

---

### Clustering
`kmeans_demo.jl` demonstrates k-means clustering on random 2D data with k=4, visualizing cluster assignments and centroids. `kmeans_cats.jl` applies k-means to a real dataset of domestic cat body and heart weights, with min-max normalization before clustering and k=3 clusters visualized with centroids overlaid.

---

### Principal Component Analysis
`pca_demo.jl` applies PCA to correlated 2D random data, reducing to 1 component and visualizing the projection and reconstruction. `pca_iris.jl` reduces the 4-feature Iris dataset to 3 principal components and plots the result as a 3D scatter grouped by species.

---

### Markov Decision Processes
Models a 4×3 grid world with stochastic transitions, a penalty state, and a goal state. Defines the full MDP (states, actions, transition function, reward function) using QuickPOMDPs.jl and solves for the optimal policy using value iteration with a discount factor of 0.95.

---

### Reinforcement Learning
Two temporal difference learning algorithms applied to a 1D grid world (7 states, LEFT/RIGHT actions, rewards at endpoints). `sarsa.jl` implements on-policy SARSA with an epsilon-greedy exploration policy. `qlearning.jl` implements off-policy Q-learning under the same setup. Both are compared against a value iteration baseline.

---

## Stack

| Area | Libraries |
|---|---|
| Regression | GLM.jl, TypedTables.jl |
| Data | CSV.jl, DataFrames.jl, RDatasets.jl, MLDatasets.jl |
| Classical ML | DecisionTree.jl, NearestNeighbors.jl, LIBSVM.jl, StatsBase.jl |
| Neural Networks | Flux.jl |
| Clustering | Clustering.jl |
| Dimensionality Reduction | MultivariateStats.jl |
| Reinforcement Learning | POMDPs.jl, QuickPOMDPs.jl, DiscreteValueIteration.jl, TabularTDLearning.jl |
| Visualization | Plots.jl |

---

## Notes

All projects were written and tested in Julia 1.12 on VS Code with the Julia extension. This repository is part of a self-directed Julia learning series — see also [Julia Foundations](https://github.com/MohammedFerdaus/Julia-Foundations) for the prerequisite analysis and statistics projects.
