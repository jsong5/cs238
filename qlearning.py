import numpy as np
import pdb
import collections

# An attempt at a tabular approach for elevators. Clearly did not work given the state space :).


class QLearning(object):
    def __init__(self, state_enum, action_enum, states, actions, rewards, next_states, discount=0.9, alpha=0.1, epochs=1000):
        self.state_enum = state_enum
        self.action_enum = action_enum
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.discount = discount
        self.alpha = alpha
        self.epochs = epochs

        pass

    def train(self):
        # qtable = np.random.rand(self.state_enum + 1, self.action_enum + 1)
        qtable = collections.defaultdict(lambda: -10)
        for _ in range(self.epochs):
            for idx in range(len(self.states)):
                state_action_pair = tuple(
                    list(self.states[idx]) + list(self.actions[idx]))

                next_state_action_list = [
                    tuple(list(self.next_states[idx]) + list([i])) for i in range(-1, 3)]
                next_state_max = max(qtable[next_state_action]
                                     for next_state_action in next_state_action_list)

                qtable[state_action_pair] += self.alpha * (self.rewards[idx] + self.discount *
                                                           next_state_max - qtable[state_action_pair])
        self.qtable = qtable

    def train_sarsa(self):
        # qtable = np.random.rand(self.state_enum + 1, self.action_enum + 1)
        qtable = KDDict(len(self.states[0]) + 1)
        for _ in range(self.epochs):
            for idx in range(len(self.states) - 1):
                state_action_pair = tuple(
                    list(self.states[idx]) + list(self.actions[idx]))

                # next_state_action_list

                next_state_action = tuple(
                    list(self.next_states[idx]) + list(self.actions[idx + 1]))

                if len(qtable) == 0:
                    qtable[next_state_action] = 0

                approximate_next_action_value = qtable[next_state_action]

                qtable[state_action_pair] += self.alpha * (self.rewards[idx] + self.discount *
                                                           approximate_next_action_value - qtable[state_action_pair])
        self.qtable = qtable

    def predict(self, state):

        max_reward = -10000
        best_action = 0

        for action in range(-1, 3):
            state_action_pair = tuple(
                list(state) + [action])
            predicted_reward = self.qtable[state_action_pair]
            if predicted_reward > max_reward:
                max_reward = predicted_reward
                best_action = action
        return [best_action]

    def predict_sarsa(self, state):

        max_reward = -10000
        best_action = 0

        for action in range(-1, 3):
            state_action_pair = tuple(
                list(state) + [action])
            predicted_reward = self.qtable[state_action_pair]
            if predicted_reward > max_reward:
                max_reward = predicted_reward
                best_action = action
        return [best_action]
