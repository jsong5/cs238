from elevator import Building
from stable_baselines3 import DQN, PPO, A2C
import gym

from elevator import scan_policy
import torch.nn as nn

import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch as th

NUM_ROLLOUTS = 200
EPISODE_LEN = 200


def train_ppo(env):
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    model.save("ppo_elevator")

    return model


def train_dqn(env):
    model = DQN("MlpPolicy", env, exploration_final_eps=0.1, verbose=0)
    model.learn(total_timesteps=100000)
    model.save("dqn_elevator")

    return model


def train_a2c(env):
    model = A2C("MlpPolicy", env, verbose=0)
    # model.learn(total_timesteps=10000)
    model.save("a2c__elevator")

    return model


def scan_rollout(env, num_eps):
    holder = []
    eps_rewards = []
    eps_avg_time = []
    people_remaining_ratio = []

    for ep in range(num_eps):
        done = False
        total_rew = 0

        while not done:
            action = scan_policy(env)
            _, reward, done, info = env.step(action)
            total_rew += reward
        env.reset()
        eps_rewards.append(total_rew)
        eps_avg_time.append(info["total_time"] / info["total_people"])
        people_remaining_ratio.append(
            info["people_remain"] / info["total_people"])
    return {"eps_rewards": eps_rewards, "eps_avg_time": eps_avg_time, "people_remaining_ratio": people_remaining_ratio, "type": "scan"}


def rollout(env, model, num_eps, model_name):
    holder = []
    state = env.reset()
    eps_rewards = []
    eps_avg_time = []
    people_remaining_ratio = []

    for ep in range(num_eps):
        done = False
        total_rew = 0

        while not done:
            action, _ = model.predict(state)
            state, reward, done, info = env.step(action)
            total_rew += reward
        state = env.reset()
        eps_rewards.append(total_rew)
        eps_avg_time.append(info["total_time"] / info["total_people"])
        people_remaining_ratio.append(
            info["people_remain"] / info["total_people"])
    return {"eps_rewards": eps_rewards, "eps_avg_time": eps_avg_time, "people_remaining_ratio": people_remaining_ratio, "type": model_name}


def make_plot(lambdas, elevators, labels):

    fig, axs = plt.subplots(len(elevators), len(labels), figsize=(20, 10))

    for elevator_idx, elevator_num in enumerate(elevators):
        data_scan = []
        data_model_ppo = []
        data_model_dqn = []
        data_model_a2c = []

        for lamb_idx, lamb_num in enumerate(lambdas):
            env = Building(10, elevator_num,
                           people_lambda=lamb_num,  max_steps=NUM_ROLLOUTS)
            model_ppo = train_ppo(env)
            model_dqn = train_dqn(env)
            model_a2c = train_a2c(env)

            data_scan.append(scan_rollout(env, NUM_ROLLOUTS))
            data_model_ppo.append(rollout(env, model_ppo, NUM_ROLLOUTS, "PPO"))
            data_model_dqn.append(rollout(env, model_dqn, NUM_ROLLOUTS, "DQN"))
            data_model_a2c.append(rollout(env, model_a2c, NUM_ROLLOUTS, "A2C"))

        for label_idx, label in enumerate(labels):
            mean_vec_scan = []
            var_vec_scan = []

            mean_vec_model_ppo = []
            var_vec_model_ppo = []

            mean_vec_model_dqn = []
            var_vec_model_dqn = []

            mean_vec_model_a2c = []
            var_vec_model_a2c = []

            for lamb_idx, lamb_num in enumerate(lambdas):
                scan_data = data_scan[lamb_idx]
                model_data_ppo = data_model_ppo[lamb_idx]
                model_data_dqn = data_model_dqn[lamb_idx]
                model_data_a2c = data_model_a2c[lamb_idx]

                mean_scan = np.mean(scan_data[label])
                var_scan = np.var(scan_data[label])

                mean_model_ppo = np.mean(model_data_ppo[label])
                mean_model_dqn = np.mean(model_data_dqn[label])
                mean_model_a2c = np.mean(model_data_a2c[label])

                var_model_ppo = np.var(model_data_ppo[label])
                var_model_dqn = np.var(model_data_dqn[label])
                var_model_a2c = np.var(model_data_a2c[label])

                mean_vec_scan.append(mean_scan)
                var_vec_scan.append(var_scan)

                mean_vec_model_ppo.append(mean_model_ppo)
                mean_vec_model_dqn.append(mean_model_dqn)
                mean_vec_model_a2c.append(mean_model_a2c)

                var_vec_model_ppo.append(var_model_ppo)
                var_vec_model_dqn.append(var_model_dqn)
                var_vec_model_a2c.append(var_model_a2c)

            mean_vec_scan = np.array(mean_vec_scan)
            var_vec_scan = np.sqrt(np.array(var_vec_scan))

            mean_vec_model_ppo = np.array(mean_vec_model_ppo)
            mean_vec_model_dqn = np.array(mean_vec_model_dqn)
            mean_vec_model_a2c = np.array(mean_vec_model_a2c)

            var_vec_model_ppo = np.sqrt(np.array(var_vec_model_ppo))
            var_vec_model_dqn = np.sqrt(np.array(var_vec_model_dqn))
            var_vec_model_a2c = np.sqrt(np.array(var_vec_model_a2c))

            ax = axs[elevator_idx, label_idx]

            # Plot for scan
            ax.errorbar(lambdas, mean_vec_scan, yerr=var_vec_scan, fmt='o-',
                        color='orange', ecolor='red', capsize=5, label="SCAN")
            ax.fill_between(lambdas, mean_vec_scan-var_vec_scan,
                            mean_vec_scan+var_vec_scan, alpha=0.2, color='black')

            # Plot for baselines ppo
            ax.errorbar(lambdas, mean_vec_model_ppo, yerr=var_vec_model_ppo, fmt='o-',
                        color='blue', ecolor='red', capsize=5, label="PPO")
            ax.fill_between(lambdas, mean_vec_model_ppo-var_vec_model_ppo,
                            mean_vec_model_ppo+var_vec_model_ppo, alpha=0.2, color='blue')

            # Plot for baselines dqn
            ax.errorbar(lambdas, mean_vec_model_dqn, yerr=var_vec_model_dqn, fmt='o-',
                        color='blue', ecolor='red', capsize=5, label="DQN")
            ax.fill_between(lambdas, mean_vec_model_dqn-var_vec_model_dqn,
                            mean_vec_model_dqn+var_vec_model_dqn, alpha=0.2, color='green')

            # # Plot for baselines a2c
            # ax.errorbar(lambdas, mean_vec_model_a2c, yerr=var_vec_model_a2c, fmt='o-',
            #             color='blue', ecolor='red', capsize=5, label="A2C")
            # ax.fill_between(lambdas, mean_vec_model_a2c-var_vec_model_a2c,
            #                 mean_vec_model_a2c+var_vec_model_a2c, alpha=0.2, color='orange')

            # Set titles
            ax.set_xlabel("Lambda Value")
            ax.set_ylabel(label)
            ax.set_title(f"Num elevators = " +
                         str(elevator_num) + " " + label)

            ax.legend()

    fig.savefig("elevators_vs.png")


if __name__ == "__main__":

    lambdas = [0.0001, 0.001, 0.01, 0.1, 0.5]
    elevators = [1, 3]

    # env = Building(10, 1, people_lambda=0.4,  max_steps=1000)
    # model = train_dqn(env)

    # model_data = rollout(env, model, 10, "dqn")
    # scan_data = scan_rollout(env, 10)

    # for data in [model_data, scan_data]:
    #     print(data["type"])
    #     print(sum(data["eps_rewards"])/len(data["eps_rewards"]))
    make_plot(lambdas, elevators, ["eps_rewards",
                                   "eps_avg_time",
                                   "people_remaining_ratio"])
