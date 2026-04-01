# K-nearest neighbors and k-d tree demo
cd(@__DIR__)
using NearestNeighbors
using Plots
using Random

gr(size = (600, 600))

# Generate random training points
Random.seed!(1)
f1_train = rand(100)
f2_train = rand(100)
x_train = permutedims([f1_train f2_train])

# Build k-d tree from training points
kdtree = KDTree(x_train)

# Random test point
f1_test = rand()
f2_test = rand()
x_test = [f1_test, f2_test]

# Find k nearest neighbors
k = 11
index_knn, distances = knn(kdtree, x_test, k, true)

# Display neighbor indices and distances
vscodedisplay([index_knn distances])

# Coordinates of nearest neighbors
f1_knn = [f1_train[i] for i in index_knn]
f2_knn = [f2_train[i] for i in index_knn]

# Plot
p = scatter(f1_train, f2_train,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "K-NN and k-d tree demo (k = $k)",
    legend = false,
    color = :blue
)

# Test point
scatter!([f1_test], [f2_test], color = :red, markersize = 10)

# Nearest neighbors
scatter!(f1_knn, f2_knn, color = :yellow, markersize = 10, alpha = 0.5)

# Lines from test point to each neighbor
for i in 1:k
    plot!([f1_test, f1_knn[i]], [f2_test, f2_knn[i]], color = :green, alpha = 0.5)
end

display(p)
savefig(p, "knn_demo.svg")