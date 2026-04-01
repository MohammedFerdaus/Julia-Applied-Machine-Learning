# Linear regression on housing price data
using CSV
using GLM
using Plots
using TypedTables

# Load data
data = CSV.File("housing_data.csv")
X = data.size
Y = round.(Int, data.price / 1000)
table_data = Table(X = X, Y = Y)

# Scatter plot
plot_scatter = scatter(X, Y,
    xlims = (0, 5000),
    ylims = (0, 800),
    xlabel = "Size (sqft)",
    ylabel = "Price (thousands of dollars)",
    title = "Housing prices in Portland",
    legend = false,
    color = :red
)

# Fit linear regression model
ols = lm(@formula(Y ~ X), table_data)

# Overlay regression line
plot!(X, predict(ols), color = :green, linewidth = 2)

# Predict price based on a new value for size
new_x_value = Table(X = [1250])
predict(ols, next_x_value)