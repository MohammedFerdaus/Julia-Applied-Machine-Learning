# Naive Bayes classifier for tennis dataset
cd(@__DIR__)
include("data_wrangling.jl")

# Feature count lookup tables per class
yes_counts = Dict(
"sunny" => n_sunny_yes + 1, "overcast" => n_overcast_yes + 1, "rainy" => n_rainy_yes + 1,
"hot" => n_hot_yes + 1, "mild" => n_mild_yes + 1, "cool" => n_cool_yes + 1,
"high" => n_high_yes + 1, "normal" => n_normal_yes + 1,
"false" => n_false_yes + 1, "true" => n_true_yes + 1
)

no_counts = Dict(
"sunny" => n_sunny_no + 1, "overcast" => n_overcast_no + 1, "rainy" => n_rainy_no + 1,
"hot" => n_hot_no + 1, "mild" => n_mild_no + 1, "cool" => n_cool_no + 1,
"high" => n_high_no + 1, "normal" => n_normal_no + 1,
"false" => n_false_no + 1, "true" => n_true_no + 1
)

# Conditional probability P(feature | class)
function likelihood(value, counts, class_total)
    return get(counts, value, 0) / class_total
end

# Generalized naive Bayes classifier
function naive_bayes(features)
    score_yes = p_yes
    score_no  = p_no

    for f in features
        score_yes *= likelihood(f, yes_counts, n_yes)
        score_no  *= likelihood(f, no_counts,  n_no)
    end

    total = score_yes + score_no
    p_yes_given_x = score_yes / total
    p_no_given_x = score_no / total
    prediction = p_yes_given_x >= p_no_given_x ? "yes" : "no"

    println("P(yes | X) = ", round(p_yes_given_x, digits = 4))
    println("P(no  | X) = ", round(p_no_given_x, digits = 4))
    println("Prediction : ", prediction)
    return prediction
end

# Predictions
println("\nInput: [sunny, hot]")
naive_bayes(["sunny", "hot"])

println("\nInput: [sunny, cool, high, true]")
naive_bayes(["sunny", "cool", "high", "true"])

println("\nInput: [overcast, mild, normal, false]")
naive_bayes(["overcast", "mild", "normal", "false"])