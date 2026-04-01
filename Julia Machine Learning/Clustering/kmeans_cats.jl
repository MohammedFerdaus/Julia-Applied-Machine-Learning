# K-means clustering — domestic cats dataset
cd(@__DIR__)
using Clustering
using Plots
using Random
using RDatasets

# Load data
cats = dataset("boot", "catsM")
vscodedisplay(cats)

# Raw scatter plot
p_raw = scatter(cats.BWt, cats.HWt,
    xlabel = "Body weight (kg)",
    ylabel = "Heart weight (g)",
    title = "Domestic cats weight data (raw)",
    legend = false
)
display(p_raw)

# Min-max normalization
f1 = cats.BWt
f2 = cats.HWt
f1_n = (f1 .- minimum(f1)) ./ (maximum(f1) - minimum(f1))
f2_n = (f2 .- minimum(f2)) ./ (maximum(f2) - minimum(f2))
x = [f1_n f2_n]'

# Cluster with k-means
k = 3
Random.seed!(1)
result = kmeans(x, k, maxiter = 100, display = :iter)
a = assignments(result)
mu = result.centers

# Plot clusters and centroids
p = scatter(f1_n, f2_n,
    xlabel = "Body weight (normalized)",
    ylabel = "Heart weight (normalized)",
    title = "Domestic cats weight data (k = $k)",
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
savefig(p, "kmeans_cats.svg")