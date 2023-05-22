import gym
import torch
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create environment
env = make_atari_env('ALE/Skiing-v5', n_envs=32, seed=0)

# # create DQN agent
# model = DQN('CnnPolicy', env, learning_rate=1e-3, buffer_size=100000, batch_size=64, learning_starts=10000,
#             exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.01, train_freq=4,
#             target_update_interval=10000, gamma=0.99, verbose=1, device=device)

# model = A2C('CnnPolicy', env, learning_rate=3e-4, n_steps=5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
#             vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-6, use_rms_prop=True, use_sde=False, sde_sample_freq=-1,
#             normalize_advantage=True, tensorboard_log=None, policy_kwargs=None, verbose=1, seed=None, device=device,
#             _init_setup_model=True)

# model = PPO('CnnPolicy', env, verbose=1, n_steps=2048, batch_size=16384, gamma=0.99,
#             learning_rate=2.5e-4, ent_coef=0.01, vf_coef=0.5, clip_range=0.2)
# load the model
# model = DQN.load('dqn_test.pt', env=env, device=device)    # DQN model
# model = A2C.load('A2C_1000000', env=env, device=device)    # A2C model
model = PPO.load('PPO_11000000.zip', env=env, device=device)    # PPO model

# train the agent
model.learn(total_timesteps=int(1e6))

# save the model
model.save("./PPO_12000000")

# evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f}")
print(f"Standard reward: {std_reward:.2f}")

# visualize trained agent
obs = env.reset()
# obs = torch.from_numpy(obs).float().to(device)

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # obs = torch.from_numpy(obs).float().to(device)
    env.render()
