# Support vector machine classifier — Iris dataset
cd(@__DIR__)
using LIBSVM
using CSV
using DataFrames
using Statistics
using Random

# Load iris dataset
using RDatasets
iris = dataset("datasets", "iris")

# Features and labels
X = Matrix(iris[:, 1:4])'  # 4 x 150 — LIBSVM expects features x samples
y = iris.Species

# Encode labels as integers
classes = unique(y)
y_encoded = [findfirst(==(label), classes) for label in y]

# Train/test split (80/20)
Random.seed!(42)
n = length(y_encoded)
idx = randperm(n)
train_idx = idx[1:120]
test_idx = idx[121:end]

X_train = X[:, train_idx]
y_train = y_encoded[train_idx]
X_test = X[:, test_idx]
y_test = y_encoded[test_idx]

# Train SVM with RBF kernel
model = svmtrain(X_train, y_train, kernel = Kernel.RadialBasis, cost = 1.0, gamma = 0.5)

# Predict on test set
y_pred, decision_values = svmpredict(model, X_test)

# Accuracy
accuracy = sum(y_pred .== y_test) / length(y_test)
println("Accuracy : ", round(accuracy * 100, digits = 2), "%")

# Confusion matrix
println("\nConfusion matrix:")
for i in 1:length(classes)
    for j in 1:length(classes)
        print(count((y_pred .== i) .& (y_test .== j)), "\t")
    end
    println()
end
println("Rows = predicted, Cols = actual")
println("Classes: ", string.(classes))

# Compare kernels
kernels = [Kernel.Linear, Kernel.RadialBasis, Kernel.Polynomial]
kernel_names = ["Linear", "RBF", "Polynomial"]

println("\nKernel comparison:")
for (k, name) in zip(kernels, kernel_names)
    m = svmtrain(X_train, y_train, kernel = k, cost = 1.0)
    preds, _ = svmpredict(m, X_test)
    acc = sum(preds .== y_test) / length(y_test)
    println("  $name : ", round(acc * 100, digits = 2), "%")
end