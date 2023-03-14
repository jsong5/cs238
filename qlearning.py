import numpy as np
import pdb
import collections


class QLearning(object):
    def __init__(self, state_enum, action_enum, states, actions, rewards, next_states, discount=0.9, alpha=0.1, epochs=30):
        # Do nothing for now.
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
        qtable = collections.defaultdict(int)
        for _ in range(self.epochs):
            for idx in range(len(self.states)):
                state_action_pair = tuple(
                    list(self.states[idx]) + list(self.actions[idx]))

                next_state_action_list = [
                    tuple(list(self.next_states[idx]) + list([i])) for i in range(3)]

                next_state_max = max(qtable[next_state_action]
                                     for next_state_action in next_state_action_list)

                qtable[state_action_pair] += self.alpha * (self.rewards[idx] + self.discount *
                                                           next_state_max - qtable[state_action_pair])
        return qtable
