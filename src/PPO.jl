
#module PPO
using Flux, Statistics, Gym
using Flux: Chain, Dense, Conv, onehot
using Gym: GymEnv, reset!, step!

LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 1e-3
GAMMA = 0.99
env = GymEnv("LunarLander-v2")
NUM_ACTIONS = env.action_space.n
DUMMY_ACTION = zeros(1, NUM_ACTIONS)
DUMMY_VALUE = zeros(1, 1)

println("Actions: ", NUM_ACTIONS)

    function exponential_average(old, new, b1)
        return old * b1 + (1-b1) * new
    end

    function proximal_policy_optimization_loss(adv, old_pred)
        function loss(y_true, y)
            prob = y_true * y
            old_prob = y_true * old_pred
            r = prob/(old_prob + 1e-10)
            r = clamp(r, 1-LOSS_CLIPPING, 1+LOSS_CLIPPING) * adv + ENTROPY_LOSS * -(prob * log(prob + 1e-10))
            return -mean(minimum((r * adv, r)))
        end
        return loss
    end
    #
    function build_actor(num_layers, input_dims, hidden_dims, num_actions)
        layers = []
        if length(input_dims) > 100
            push!(layers, Conv((4, 4), input_dims[3]=>8, relu))
            push!(layers, Conv((4, 4), 8=>16, relu))
            push!(layers, x -> reshape(x, :, size(x, 4)))
            push!(layers, x -> Dense(size(x)[1], hidden_dims, relu))
        else
            push!(layers, Dense(input_dims[1], hidden_dims, relu))
        end
        for i in 1:num_layers-1
            push!(layers, Dense(hidden_dims, hidden_dims, relu))
        end
        push!(layers, Dense(hidden_dims, num_actions, tanh))
        # push!(layers, x -> tanh(x))
        println(layers)
        return Chain(layers...)
    end
    #
    function build_critic(num_layers, input_dims, hidden_dims)
        layers = []
        push!(layers, Dense(input_dims, hidden_dims, tanh))
        for i in 1:num_layers-1
            push!(layers, Dense(hidden_dims, tanh))
        end
        push!(layers, Dense(hidden_dims, 1))
        return Chain(layers...)
    end

    function reset_env!(env, episode_count)
        episode_count += 1
        reset(env)
    end

    # actor = build_actor(3, env.observation_space.shape, 64, NUM_ACTIONS)
    # critic = build_critic()
    # a_loss(x, y) =
    # c_loss(x, y) = Flux.mse(critic(x), y)
    # a_opt = Flux.ADAM()
    # c_opt = Flux.ADAM()
    # reward_tracker = []
    # rewards = []

    function get_action(actor, obs)
        p = actor(obs)
        action = round(argmax(p))
        return action, onehot(action, 1:NUM_ACTIONS), p
    end

    function main()
        actor = build_actor(3, env.observation_space.shape, 64, NUM_ACTIONS)
        critic = build_critic(3, env.observation_space.shape, 64)
        batch = [[], [], [], []]
        reward_writer = []
        for i in 1:10
            temp_batch = [[], [], []]
            rewards = []
            state = reset!(env)
            r = 0
            action = sample(env.action_space)
            state, reward, done , _ = step!(env, action)
            println("Sample Rewards: ", reward, " Action: ", action, " Done: ", done)
            while !done
                action, action_matrix, predicted_action = get_action(actor, state)
                println("Rewards: ", reward, " Action: ", action)
                push!(rewards, reward)
                push!(temp_batch[1], state)
                push!(temp_batch[2], action_matrix)
                push!(temp_batch[3], predicted_action)
                state, reward, done, _ = step!(env, action)
                r += reward
            end
            for j in length(rewards)-2:-1:1
                rewards[j] += rewards[j+1] * GAMMA
            end
            sum = 0
            for j in 1:length(rewards)
                sum += rewards[i]
            end
            push!(reward_writer, sum)
            push!(batch[1], temp_batch[1])
            push!(batch[2], temp_batch[2])
            push!(batch[3], temp_batch[3])
            push!(batch[4], rewards)
        end
        actor_opt = ADAM()
        critic_opt = ADAM()
        actor_loss_fn(adv, old_pred) = proximal_policy_optimization_loss(adv, old_pred)(actor())
        critic_loss_fn(x, y) = mse(actor(x), y)
        Flux.train!()
    end

    main()
#end
