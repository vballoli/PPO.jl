using OpenAIGym

env = GymEnv(:CartPole, :v0)
for i ∈ 1:20
  T = 0
  R = 0
  for i in 1:10
    r, s_ = OpenAIGym.step!(env, length(actions(env, OpenAIGym.reset!(env))))
    R += r
  end
  # R = run_episode(env, RandomPolicy()) do (s, a, r, s′)
  #   OpenAIGym.step!()
  #   T += 1
  # end
  @info("Episode $i finished after $T steps. Total reward: $R")
end
