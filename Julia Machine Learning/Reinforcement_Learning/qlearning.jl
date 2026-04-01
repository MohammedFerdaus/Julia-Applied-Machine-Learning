# Q-learning temporal difference learning — 1D grid world
cd(@__DIR__)
using POMDPs
using POMDPModelTools
using QuickPOMDPs
using DiscreteValueIteration
using TabularTDLearning
using POMDPPolicies
using Random

# State and action types
struct State
    x::Int
end

@enum Action LEFT RIGHT

# State and action spaces
null = State(-1)
S = vcat(vec([State(x) for x = 1:7]), null)
A = [LEFT, RIGHT]

# Movement deltas
const MOVEMENTS = Dict(
    LEFT => State(-1),
    RIGHT => State(1)
)

Base.:+(s1::State, s2::State) = State(s1.x + s2.x)

# Reward function
function R(s, a = missing)
    if s == State(1)
        return -1
    elseif s == State(7)
        return 1
    end
    return 0
end

# Transition function
function T(s::State, a::Action)
    if R(s) != 0
        return Deterministic(null)
    end

    len_a = length(A)
    next_states = Vector{State}(undef, len_a + 1)
    probabilities = zeros(len_a + 1)

    for (index, a_prime) in enumerate(A)
        prob = (a_prime == a) ? 0.8 : 0.2
        dest = s + MOVEMENTS[a_prime]
        next_states[index + 1] = dest

        if 1 <= dest.x <= 7
            probabilities[index + 1] += prob
        end
    end

    next_states[1] = s
    probabilities[1] = 1 - sum(probabilities)

    return SparseCat(next_states, probabilities)
end

# Termination condition
termination(s::State) = s == null

abstract type GridWorld <: MDP{State, Action} end

# Value iteration baseline
mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = 0.95,
    isterminal = termination
)

solver = ValueIterationSolver(max_iterations = 30)
policy = solve(solver, mdp)

# Q-learning MDP with initial state
q_mdp = QuickMDP(GridWorld,
    states = S,
    actions = A,
    transition = T,
    reward = R,
    discount = 0.95,
    initialstate = State(4),
    isterminal = termination
)

# Q-learning solver
Random.seed!(1)
q_solver = QLearningSolver(
    n_episodes = 20,
    learning_rate = 0.9,
    exploration_policy = EpsGreedyPolicy(q_mdp, 0.5),
    verbose = false
)
q_policy = solve(q_solver, q_mdp)