# PCA applied to Iris dataset — 3D visualization
cd(@__DIR__)
using MultivariateStats
using Plots
using RDatasets

# Load and preprocess Iris data
iris = dataset("datasets", "iris")
x = Matrix(iris[:, 1:4])'
y = Vector{String}(iris.Species)
species = reshape(unique(iris.Species), (1, 3))

# Fit PCA and reduce to 3 components
model = fit(PCA, x, maxoutdim = 3)
x_transform = MultivariateStats.transform(model, x)

pc1 = x_transform[1, :]
pc2 = x_transform[2, :]
pc3 = x_transform[3, :]

# 3D scatter plot of principal components
gr(size = (640, 480))
p = scatter(pc1, pc2, pc3,
    xlabel = "PC1",
    ylabel = "PC2",
    zlabel = "PC3",
    markersize = 1,
    group = y,
    label = species
)

display(p)
savefig(p, "pca_iris.svg")