# K-nearest neighbors classifier — Iris dataset
cd(@__DIR__)
using NearestNeighbors
using Plots
using Random
using RDatasets
using StatsBase
using Statistics

# Load Iris dataset
iris = dataset("datasets", "iris")
x = Matrix(iris[:, 1:4])
y = Vector{String}(iris.Species)

# Per-class stratified sampling
function perclass_splits(y, percent)
    keep_index = []
    for class in unique(y)
        class_index = findall(y .== class)
        row_index = randsubseq(class_index, percent)
        push!(keep_index, row_index)
    end
    return vcat(keep_index...)
end

# Train/test split (67% train, 33% test)
Random.seed!(1)
index_train = perclass_splits(y, 0.67)
index_test = setdiff(1:length(y), index_train)

x_train = x[index_train, :]
x_test = x[index_test, :]
y_train = y[index_train]
y_test = y[index_test]

# Transpose for NearestNeighbors (features x samples)
x_train_t = permutedims(x_train)
x_test_t = permutedims(x_test)

# Build k-d tree and find k nearest neighbors
k = 5
kdtree = KDTree(x_train_t)
index_knn, distances = knn(kdtree, x_test_t, k, true)

# Map neighbor indices to class labels
index_knn_matrix = permutedims(hcat(index_knn...))
knn_classes = y_train[index_knn_matrix]
vscodedisplay(knn_classes)

# Majority vote prediction
y_hat = [argmax(countmap(knn_classes[i, :])) for i in 1:length(y_test)]

# Accuracy
accuracy = mean(y_hat .== y_test)
println("Accuracy : ", round(accuracy * 100, digits = 2), "%")

# Prediction breakdown
check = [y_hat[i] == y_test[i] for i in 1:length(y_hat)]
check_display = [y_hat y_test check]
vscodedisplay(check_display)