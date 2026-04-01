# Multilayer perceptron classifier — MNIST dataset
cd(@__DIR__)
using Flux
using Flux: crossentropy, onecold, onehotbatch
using MLDatasets
using Plots
using Random
using Statistics

Random.seed!(1)

# Load MNIST data
train_data = MLDatasets.MNIST(split=:train)[:]
test_data = MLDatasets.MNIST(split=:test)[:]

x_train_raw = Float32.(train_data.features)
y_train_raw = train_data.targets
x_test_raw = Float32.(test_data.features)
y_test_raw = test_data.targets

# Flatten images from 28x28 to 784-element vectors
x_train = Flux.flatten(x_train_raw)
x_test = Flux.flatten(x_test_raw)

# One-hot encode labels (digits 0-9)
y_train = onehotbatch(y_train_raw, 0:9)
y_test = onehotbatch(y_test_raw, 0:9)

# Multilayer perceptron: 784 inputs → 32 hidden (ReLU) → 10 outputs (softmax)
model = Chain(
    Dense(28 * 28, 32, relu),
    Dense(32, 10),
    softmax
)

# ADAM optimizer and state
opt = Adam(0.01)
opt_state = Flux.setup(opt, model)

# Loss function
loss(m, x, y) = crossentropy(m(x), y)

# Training loop
loss_history = []
epochs = 500

for epoch in 1:epochs
    Flux.train!(loss, model, [(x_train, y_train)], opt_state)
    train_loss = loss(model, x_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch $epoch : loss = ", round(train_loss, digits = 4))
end

# Predictions and accuracy
y_hat = onecold(model(x_test)) .- 1
accuracy = mean(y_hat .== y_test_raw)
println("Test accuracy : ", round(accuracy * 100, digits = 2), "%")

# Prediction breakdown
check = [y_hat[i] == y_test_raw[i] for i in 1:length(y_hat)]
check_display = [collect(1:length(y_hat)) y_hat y_test_raw check]
vscodedisplay(check_display)

# Learning curve
gr(size = (600, 400))
p = plot(1:epochs, loss_history,
    xlabel = "Epoch",
    ylabel = "Loss",
    title = "Learning curve",
    legend = false,
    color = :blue,
    linewidth = 1
)

display(p)
savefig(p, "learning_curve.svg")
