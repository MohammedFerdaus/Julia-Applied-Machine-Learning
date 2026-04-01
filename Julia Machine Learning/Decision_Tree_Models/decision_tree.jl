# Decision tree classifier — Iris dataset
cd(@__DIR__)
using DecisionTree
using Random
using Statistics

# Load data
x, y = load_data("iris")
x = float.(x)
y = string.(y)

# Per-class stratified sampling for train/test split
function perclass_splits(y, percent)
    keep_index = []
    for class in unique(y)
        class_index = findall(y .== class)
        row_index = randsubseq(class_index, percent)
        push!(keep_index, row_index)
    end
    return vcat(keep_index...)
end

# Train/test split (67% train, 33% test)
Random.seed!(1)
train_index = perclass_splits(y, 0.67)
test_index = setdiff(1:length(y), train_index)

x_train = x[train_index, :]
x_test = x[test_index, :]
y_train = y[train_index]
y_test = y[test_index]

# Train decision tree
model = DecisionTreeClassifier(max_depth = 2)
fit!(model, x_train, y_train)

# Print tree structure
print_tree(model)

# Predictions and accuracy
y_hat = predict(model, x_test)
accuracy = mean(y_hat .== y_test)
println("Accuracy : ", round(accuracy * 100, digits = 2), "%")

# Confusion matrix
println("\nConfusion matrix:")
println(confusion_matrix(y_test, y_hat))

# Prediction breakdown
println("\nPrediction check (predicted | actual | correct):")
check = [y_hat[i] == y_test[i] for i in 1:length(y_hat)]
check_display = [y_hat y_test check]
vscodedisplay(check_display)

# Class probabilities for each prediction
println("\nClass probabilities:")
prob = predict_proba(model, x_test)
vscodedisplay(prob)