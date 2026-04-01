# Tennis dataset — data wrangling for naive Bayes classifier
cd(@__DIR__)
using CSV
using DataFrames

# Load data
df = DataFrame(CSV.File("tennis.csv"))

# Feature and label columns
x1 = df.outlook # sunny, overcast, rainy
x2 = df.temp # hot, mild, cool
x3 = df.humidity # high, normal
x4 = df.windy # false, true
y = df.play # yes, no

# Class counts and prior probabilities
n = length(y)
n_yes = count(==("yes"), y)
n_no = count(==("no"),  y)
p_yes = n_yes / n
p_no = n_no / n

# Split by class
df_yes = df[df.play .== "yes", :]
df_no  = df[df.play .== "no",  :]

# Feature counts for class = yes
n_sunny_yes = count(==("sunny"), df_yes.outlook)
n_overcast_yes = count(==("overcast"), df_yes.outlook)
n_rainy_yes = count(==("rainy"), df_yes.outlook)

n_hot_yes = count(==("hot"), df_yes.temp)
n_mild_yes = count(==("mild"), df_yes.temp)
n_cool_yes = count(==("cool"), df_yes.temp)

n_high_yes = count(==("high"), df_yes.humidity)
n_normal_yes = count(==("normal"), df_yes.humidity)

n_false_yes = count(==("false"), string.(df_yes.windy))
n_true_yes = count(==("true"), string.(df_yes.windy))

# Feature counts for class = no
n_sunny_no = count(==("sunny"), df_no.outlook)
n_overcast_no = count(==("overcast"), df_no.outlook)
n_rainy_no = count(==("rainy"), df_no.outlook)

n_hot_no = count(==("hot"), df_no.temp)
n_mild_no = count(==("mild"), df_no.temp)
n_cool_no = count(==("cool"), df_no.temp)

n_high_no = count(==("high"), df_no.humidity)
n_normal_no = count(==("normal"), df_no.humidity)

n_false_no = count(==("false"), string.(df_no.windy))
n_true_no = count(==("true"), string.(df_no.windy))
