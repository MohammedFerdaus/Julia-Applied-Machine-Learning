# PCA demo — random correlated data
cd(@__DIR__)
using MultivariateStats
using Plots
using Random

Random.seed!(1)

# Generate correlated data
f0 = collect(0.05:0.05:5)
f1 = f0 .+ rand(100)
f2 = f0 .+ rand(100)
x = [f1 f2]'

gr(size = (480, 480))

# Raw data plot
p_raw = scatter(f1, f2,
    xlabel = "Feature 1",
    ylabel = "Feature 2",
    title = "Random correlated data",
    legend = false
)
display(p_raw)

# Fit PCA model and reduce to 1 component
model = fit(PCA, x, maxoutdim = 1)
x_transform = MultivariateStats.transform(model, x_transform)
x_reconstruct = MultivariateStats.reconstruct(model, x_transform)

# Plot transformed data (1D projection)
p_transform = scatter(x_transform', zeros(100),
    xlabel = "PC1",
    title = "PCA projection (1 component)",
    legend = false,
    color = :red,
    alpha = 0.5
)
display(p_transform)

# Overlay reconstructed points on raw data
scatter!(p_raw,
    x_reconstruct[1, :],
    x_reconstruct[2, :],
    color = :red,
    alpha = 0.5
)
display(p_raw)

savefig(p_raw, "pca_demo.svg")