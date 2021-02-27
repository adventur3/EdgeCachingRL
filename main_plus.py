# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from RL_brain import DeepQNetwork
from caching_env_plus import CachingEnv
import numpy as np

def run(env, RL):
    total_step = 0
    #reward_his = []
    cost_his=[]
    cache_hit_ratio_his = []
    for episode in range(300):
        observation = env.reset()
        episodeRequestCount = 0
        episodeCacheHitCount = 0
        costSum = 0
        for step in range(2000):
            action = RL.choose_action(observation)
            observation_, reward, currentRequestCount, hitCacheCount = env.step(action)
            # if episode >= 290 and step >= 1900:
            #     episodeRequestCount += currentRequestCount
            #     episodeCacheHitCount += hitCacheCount
            # if episode >= 290 and step == 1999:
            #     reward_his.append(reward)
            RL.store_transition(observation, action, reward, observation_)
            if (total_step > 200) and (total_step % 5 == 0):
                RL.learn()
            observation = observation_
            total_step += 1
            episodeCacheHitCount += hitCacheCount
            episodeRequestCount += currentRequestCount
            costSum += -reward
        #reward_his.append(env.rsu_residual_capcity[3])
        # if episode >= 290:
        #     cache_hit_ratio_his.append(episodeCacheHitCount/episodeRequestCount)
    #plot_reward(reward_his)
        cache_hit_ratio_his.append(episodeCacheHitCount/episodeRequestCount)
        cost_his.append(costSum)
    plot_cache_hit_ratio(cache_hit_ratio_his)
    plot_cost(cost_his)
    print(cache_hit_ratio_his)
    print(cost_his)
    print(sum(cache_hit_ratio_his)/len(cache_hit_ratio_his))
    print(sum(cost_his) / len(cost_his))
    # for i in range(len(reward_his)):
    #     print(reward_his[i])

def plot_cache_hit_ratio(cache_hit_ratio_his):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(cache_hit_ratio_his)), cache_hit_ratio_his)
    plt.ylabel('Cache Hit Ratio')
    plt.xlabel('Episodes')
    plt.show()

def plot_cost(cost_his):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(cost_his)), cost_his)
    plt.ylabel('Cost')
    plt.xlabel('Episodes')
    plt.show()

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
