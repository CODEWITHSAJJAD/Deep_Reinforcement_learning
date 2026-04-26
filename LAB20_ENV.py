# import gym
# env=gym.make('LunarLander-v2',render_mode='human')
# observation,info=env.reset()
# print('observation',observation)
# print('information',info)
# episode_over=False
# while not episode_over:
#     action=env.action_space.sample()#action taken under policy
#     observation,reward,terminated,truncated,info=env.step(action)
#     episode_over=terminated  or truncated
# env.close()
# import gym
# env=gym.make('MountainCar-v0www',render_mode='human')
# observation,info=env.reset()
# print('observation',observation)
# print('information',info)
# episode_over=False
# while not episode_over:
#     action=env.action_space.sample()#action taken under policy
#     observation,reward,terminated,truncated,info=env.step(action)
#     episode_over=terminated  or truncated
# env.close()
# import gym
# env=gym.make('FrozenLake-v1',render_mode='human')
# observation,info=env.reset()
# print('observation',observation)
# print('information',info)
# episode_over=False
# while not episode_over:
#     action=env.action_space.sample()#action taken under policy
#     observation,reward,terminated,truncated,info=env.step(action)
#     episode_over=terminated  or truncated
# env.close()
# import gym
# env=gym.make('FrozenLake8x8-v1',render_mode='human')
# observation,info=env.reset()
# print('observation',observation)
# print('information',info)
# episode_over=False
# while not episode_over:
#     action=env.action_space.sample()#action taken under policy
#     observation,reward,terminated,truncated,info=env.step(action)
#     episode_over=terminated  or truncated
# env.close()
#############################
import gym
from gym.wrappers import RecordEpisodeStatistics,RecordVideo
num_eval_episodes=4
env=gym.make('CartPole-v1',render_mode='rgb_array')
env=RecordVideo(env,video_folder='cartpol_agent',name_prefix='eval',episode_trigger=lambda x:True)
env=RecordEpisodeStatistics(env)
for episode_num in range(num_eval_episodes):
    observation,info=env.reset()
    episode_over=False
    while not episode_over:
        action=env.action_space.sample()
        observation,reward,terminated,truncated,info=env.step(action)
        episode_over=terminated or truncated
env.close()
print("Episode total rewards:",env.return_queue)
print("Episode lengths:",env.length_queue)
