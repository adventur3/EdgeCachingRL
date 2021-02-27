# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from RL_brain import DeepQNetwork
from caching_env_popularFirst import CachingEnv
import numpy as np

def run(env, RL):
    total_step = 0
    reward_his = []
    cache_hit_ratio_his = []
    observation = env.reset()
    for episode in range(300):
        episodeRequestCount = 0
        episodeCacheHitCount = 0
        observation_, reward, currentRequestCount, hitCacheCount = env.step(0)
        if episode >= 290:
            episodeRequestCount += currentRequestCount
            episodeCacheHitCount += hitCacheCount
        if episode >= 290:
            reward_his.append(reward)
        if episode >= 290:
            cache_hit_ratio_his.append(episodeCacheHitCount/episodeRequestCount)
    #plot_reward(reward_his)
    plot_reward(cache_hit_ratio_his)
    plot_reward(reward_his)
    print(cache_hit_ratio_his)
    print(sum(cache_hit_ratio_his)/len(cache_hit_ratio_his))
    print(reward_his)
    #print(env.cache_state)
    # for i in range(len(reward_his)):
    #     print(reward_his[i])

def plot_reward(reward_his):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(reward_his)), reward_his)
    plt.ylabel('Reward')
    plt.xlabel('Steps')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = CachingEnv()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run(env, RL)
    RL.plot_cost()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
