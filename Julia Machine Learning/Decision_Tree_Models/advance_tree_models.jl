# Random forest and AdaBoost classifiers — Iris dataset
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

# Helper to evaluate and print results for any model
function evaluate(model, x_test, y_test, name)
    y_hat = predict(model, x_test)
    accuracy = mean(y_hat .== y_test)
    println("\n$name")
    println("Accuracy : ", round(accuracy * 100, digits = 2), "%")
    println("Confusion matrix:")
    println(confusion_matrix(y_test, y_hat))
    return y_hat
end

# Random forest
rf = RandomForestClassifier(n_trees = 20)
fit!(rf, x_train, y_train)
y_hat_rf = evaluate(rf, x_test, y_test, "Random forest")

# AdaBoost
ada = AdaBoostStumpClassifier(n_iterations = 20)
fit!(ada, x_train, y_train)
y_hat_ada = evaluate(ada, x_test, y_test, "AdaBoost")

# Side by side comparison
println("\nPrediction comparison (RF | AdaBoost | actual):")
comparison = [y_hat_rf y_hat_ada y_test]
vscodedisplay(comparison)

# Class probabilities
println("\nRandom forest class probabilities:")
vscodedisplay(predict_proba(rf, x_test))

println("\nAdaBoost class probabilities:")
vscodedisplay(predict_proba(ada, x_test))