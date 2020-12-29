# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
