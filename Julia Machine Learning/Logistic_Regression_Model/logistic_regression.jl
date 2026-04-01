# Logistic regression via gradient descent
cd(@__DIR__)
using Plots
using CSV

# Load and clean wolf spider data
data = CSV.File("wolfspider.csv")
X = Float64.(data.feature)
Y = [v == "present" ? 1.0 : 0.0 for v in data.class]

# Base scatter plot
p = scatter(X, Y,
    xlabel = "Size of grains of sand (mm)",
    ylabel = "Presence (0 = absent, 1 = present)",
    title = "Wolf spider presence classifier (epochs = 0)",
    legend = false,
    color = :red,
    markersize = 5
)

# Initialize parameters
theta_0 = 0.0
theta_1 = 1.0
m = length(X)

# Parameter and cost history
t0_history = [theta_0]
t1_history = [theta_1]
J_history = Float64[]

# Hypothesis functions
z(x) = theta_0 .+ theta_1 .* x
h(x) = 1 ./ (1 .+ exp.(-z(x)))

# Binary cross-entropy cost
function cost(y_hat)
    (-1 / m) * sum(Y .* log.(y_hat .+ 1e-10) .+
    (1 .- Y) .* log.(1 .- y_hat .+ 1e-10))
end

# Gradient descent loop
n_epochs = 100
alpha = 0.1

for i in 1:n_epochs
    y_hat = h(X)

    d_theta_0 = (1 / m) * sum(y_hat .- Y)
    d_theta_1 = (1 / m) * sum((y_hat .- Y) .* X)

    global theta_0 -= alpha * d_theta_0
    global theta_1 -= alpha * d_theta_1

    push!(t0_history, theta_0)
    push!(t1_history, theta_1)
    push!(J_history, cost(h(X)))

    plot!(p, 0:0.01:1.2, h,
        color = :blue,
        alpha = 0.025,
        title = "Wolf spider presence classifier (epochs = $i)"
    )
end

# Final hypothesis overlay
plot!(p, 0:0.01:1.2, h, color = :green, linewidth = 2)
display(p)

# Cost curve
p_cost = plot(1:n_epochs, J_history,
    xlabel = "Epoch",
    ylabel = "Cost",
    title = "Cost over training",
    legend = false,
    color = :blue,
    linewidth = 2
)
display(p_cost)

println("Final parameters:")
println("  theta_0 : ", round(theta_0, digits=4))
println("  theta_1 : ", round(theta_1, digits=4))
println("  Final cost : ", round(J_history[end], digits=4))
