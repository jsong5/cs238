import numpy as np
import pdb
import collections


class OnlineQLearning(object):
    def __init__(self, actions, step_function, discount=0.9, alpha=0.1, epochs=100):
