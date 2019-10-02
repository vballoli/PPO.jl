using using Flux, Statistics, Gym
using Flux: Chain, Dense, Conv, onehot
using Gym: GymEnv, reset!, step!

actions = []
states = []
logprobs = []
rewards = []
is_terminals = []

function clear_memory()
    actions = []
    states = []
    logprobs = []
    rewards = []
    is_terminals = []
end

function init_actor_critic(state_dim, action_dim, n_latent_var)
    global affine = Dense(state_dim, n_latent_var)
    global actor = Chain(
        Dense(state_dim, n_latent_var, tanh),
        Dense(n_latent_var, n_latent_var, tanh),
        Dense(n_latent_var, action_dim),
        softmax
    )
    global critic = Chain(
        Dense(state_dim, n_latent_var, tanh),
        Dense(n_latent_var, n_latent_var, tanh),
        Dense(n_latent_var, 1)
    )
end

function act(state, memory)
    global actor, critic
    action_probs = actor(state)
    dist = Categorical(action_probs)
    action = sample(dist.support)

    push!(states, state)
    push!(actions, action)
    push!(logprobs, logrob)

    return action
end

function evaluate(state, action)
    global actor, critic
    action_probs = actor(state)
    dist = Categorical(action_probs)
    action_logprobs = sample(dist.support)
    dist_entropy = entropy(dist)
    state_value = critic(state)
    return [action_logprobs, dropdims(state_value, dims=(findall(size(state_value) .== 1)...,)), dist_entropy]
end
