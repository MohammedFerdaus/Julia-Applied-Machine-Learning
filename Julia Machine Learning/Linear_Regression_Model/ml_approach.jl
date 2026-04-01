# Add a directory
cd(@__DIR__)

# Linear regression — iterative gradient descent vs GLM
using CSV
using GLM
using Plots
using TypedTables
using Statistics

# Load data
data = CSV.File("housing_data.csv")
X = Float64.(data.size)
Y = round.(Float64, data.price / 1000)
table_data = Table(X = X, Y = Y)

# Normalize X for numerical stability
X_mean = mean(X)
X_std  = std(X)
X_norm = (X .- X_mean) ./ X_std

# Initialize parameters
theta_0 = 0.0
theta_1 = 0.0
m = length(X)

# Hypothesis and cost functions
h(x) = theta_0 .+ theta_1 .* x
cost(y_hat, Y) = (1 / (2m)) * sum((y_hat .- Y).^2)

# Gradient descent hyperparameters
alpha = 0.01
n_epochs = 1000
J_history = Float64[]

# Gradient descent loop
for epoch in 1:n_epochs
    y_hat = h(X_norm)
    J = cost(y_hat, Y)
    push!(J_history, J)

    d_theta_0 = (1 / m) * sum(y_hat .- Y)
    d_theta_1 = (1 / m) * sum((y_hat .- Y) .* X_norm)

    global theta_0 -= alpha * d_theta_0
    global theta_1 -= alpha * d_theta_1
end

println("Gradient descent parameters:")
println("  theta_0 (intercept) : ", round(theta_0, digits=4))
println("  theta_1 (slope)     : ", round(theta_1, digits=4))

# GLM model for comparison
ols = lm(@formula(Y ~ X), table_data)
ols_pred = predict(ols)

# Gradient descent predictions (denormalized back to original X scale)
gd_pred = h(X_norm)

# Accuracy comparison (R^2 and RMSE)
function r_squared(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - ss_res / ss_tot
end

function rmse(y_true, y_pred)
    return sqrt(mean((y_true .- y_pred).^2))
end

println("\nModel comparison:")
println("  GLM R^2  : ", round(r_squared(Y, ols_pred), digits=4))
println("  GD  R^2  : ", round(r_squared(Y, gd_pred),  digits=4))
println("  GLM RMSE : ", round(rmse(Y, ols_pred), digits=4))
println("  GD  RMSE : ", round(rmse(Y, gd_pred),  digits=4))

# Plot 1 (data and both regression lines)
X_norm_range = LinRange(minimum(X_norm), maximum(X_norm), 300)
gd_line = theta_0 .+ theta_1 .* X_norm_range
X_real_range = X_norm_range .* X_std .+ X_mean
ols_line = coef(ols)[1] .+ coef(ols)[2] .* X_real_range

p1 = scatter(X, Y,
    xlabel = "Size (sqft)",
    ylabel = "Price (thousands)",
    title = "Gradient descent vs GLM (epochs = $n_epochs)",
    label = "Data",
    color = :red,
    alpha = 0.6
)
plot!(p1, X_real_range, gd_line, label = "Gradient descent", color = :blue, linewidth = 2)
plot!(p1, X_real_range, ols_line, label = "GLM", color = :green, linewidth = 2)

# Plot 2 (cost over epochs)
p2 = plot(1:n_epochs, J_history,
    xlabel = "Epoch",
    ylabel = "Cost",
    title = "Cost over training",
    legend = false,
    color = :blue,
    linewidth = 2
)

# Display both plots
plot(p1, p2, layout = (1, 2), size = (900, 400))

display(p1)
savefig(p1, "regression_comparison.svg")

display(p2)
savefig(p2, "cost_curve.svg")