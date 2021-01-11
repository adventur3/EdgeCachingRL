# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from RL_brain import DeepQNetwork
from caching_env import CachingEnv

def run(env, RL):
    total_step = 0
    for episode in range(300):
        observation = env.reset()
        for step in range(1000):
            action = RL.choose_action(observation)
            observation_, reward = env.step(action)
            RL.store_transition(observation, action, reward, observation_)

            if (total_step > 200) and (total_step % 5 == 0):
                RL.learn()
            observation = observation_
            total_step += 1

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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
