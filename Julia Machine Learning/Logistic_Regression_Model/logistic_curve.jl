# Logistic curve visualization
cd(@__DIR__)
using Plots
using CSV

# Logistic (sigmoid) function
logistic(x) = 1 / (1 + exp(-x))

# Plot base sigmoid curve
p_logistic = plot(-6:0.01:6, logistic,
    xlabel = "Input value (x)",
    ylabel = "Output value (y)",
    title = "Logistic (sigmoid) curve",
    legend = false,
    color = :blue
)

# Modified sigmoid with learnable parameters
theta_0 = 1.0
theta_1 = 1.0

z(x) = theta_0 .+ theta_1 .* x
h(x) = 1 ./ (1 .+ exp.(-z(x)))

plot!(p_logistic, -6:0.01:6, h, color = :red, linestyle = :dash)
display(p_logistic)

# Load and clean wolf spider data
data = CSV.File("wolfspider.csv")
X = Float64.(data.feature)
Y = [v == "present" ? 1.0 : 0.0 for v in data.class]

# Scatter plot of raw data
p_data = scatter(X, Y,
    xlabel = "Size of grains of sand (mm)",
    ylabel = "Presence (0 = absent, 1 = present)",
    title = "Wolf spider presence classifier",
    legend = false,
    color = :red,
    markersize = 5
)
display(p_data)