# K-means clustering demo — random data
cd(@__DIR__)
using Clustering
using Plots
using Random

gr(size = (600, 600))
Random.seed!(1)

# Generate random data
f1 = rand(100)
f2 = rand(100)
x = [f1 f2]'

# Cluster with k-means
k = 4
result = kmeans(x, k, maxiter = 100, display = :iter)
a = assignments(result)
mu = result.centers

# Plot clusters and centroids
p = scatter(f1, f2,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "K-means clustering demo (k = $k)",
    legend = false,
    group = a,
    markersize = 10,
    alpha = 0.7
)

scatter!(mu[1, :], mu[2, :],
    color = :yellow,
    markersize = 20,
    alpha = 0.7
)

display(p)
savefig(p, "kmeans_demo.svg")