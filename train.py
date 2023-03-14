from qlearning import QLearning
from elevator import *

if __name__ == "__main__":
    global lobby
    people_gen = [PersonGenerator(floor, parameter_tuple=((1)))
                  for floor in range(3)]
    simulation = Building(people_gen, 1)
    actions = [ELEVATOR_STOP for _ in range(10)]

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    simulation.lobby_holder[0].add(person1)
    simulation.lobby_holder[0].add(person2)
    simulation.lobby_holder[0].add(person3)

    found_policy = []

    batch_reward_episodes = []
    batch_policies = []
    batch_states = []
    batch_data = []

    for train_samples in range(10):
        policy = []
        reward_vec = []
        states = []
        state = [False, False, False, False, False, False]
        for i in range(50):
            # Dont queue past 30 iterations.
            if i < 30:
                for idx, generator in enumerate(people_gen):
                    for person in generator.sample():
                        simulation.lobby_holder[0].add(person)
            old_state = state

            state = simulation.simple_elevator_local_state(
                simulation.elevators[0])
            states.append(state)
            policy = scan_policy(simulation)
            reward = simulation.step(policy)
            batch_data.append(np.array([old_state, policy, reward, state]))

            reward_vec.append(reward)
            found_policy.append(policy)

        simulation.reset()
        batch_policies.append(policy)
        batch_reward_episodes.append(reward_vec)
        batch_states.append(states)

    batch_reward_episodes = np.array(batch_reward_episodes)
    batch_policies = np.array(batch_policies)
    batch_states = np.array(batch_states)
    batch_data = np.array(batch_data)
    input_batch_data = batch_data.T

    learner = QLearning(
        None, None, input_batch_data[0], input_batch_data[1], input_batch_data[2], input_batch_data[3])
    table = learner.train()
    # pdb.set_trace()
