from elevator import Building
from stable_baselines3 import DQN
from stable_baselines3 import PPO

from elevator import scan_policy

import pdb
import numpy as np


def train_dqn(env):
    print("Initializing env")

    print("Initializing model")
    # model = DQN("MlpPolicy", env, verbose=1,
    #             learning_rate=0.001, exploration_final_eps=0.1)
    # model = DQN("MlpPolicy", env, verbose=1)
    model = PPO("MlpPolicy", env, verbose=1)

    print("Training model")
    model.learn(total_timesteps=250000)

    print("Training done and saved")
    model.save("deepq_elevator")


def scan(env):

    holder = []
    env.reset()

    for ep in range(100):
        done = False
        total_rew = 0
        reward = 0

        while not done:
            action = scan_policy(env)
            # action_temp = np.random.randint(3)
            # action = action_temp[0] + 1
            # print(action)
            # env.render()
            _, reward, done, info = env.step(action)
            # env.render()
            total_rew += reward

        env.reset()
        holder.append(total_rew)
    print("Scan reward ", sum(holder)/len(holder))

    return sum(holder)/len(holder)


if __name__ == "__main__":
    env = Building(3, 1, max_steps=20)

    train_dqn(env)
    scan(env)
