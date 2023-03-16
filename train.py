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

    for train_samples in range(1000):
        policy = []
        reward_vec = []
        states = []
        # state = [False, False, False, False, False, False]
        for i in range(10):
            # Dont queue past x iterations.
            if i < 3:
                for idx, generator in enumerate(people_gen):
                    for person in generator.sample():
                        simulation.lobby_holder[0].add(person)
            # state = simulation.simple_elevator_local_state(
            #     simulation.elevators[0])
            state = simulation._next_observation()

            states.append(state)
            policy = scan_policy(simulation)
            # if np.random.rand() > 0.0:
            #     policy = np.random.choice([-1, 0, 1, 2], 1)

            reward = simulation.step(policy)
            # simulation.render()
            batch_data.append(np.array([state, policy, reward]))

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

    np.save("./temp.npy", input_batch_data)
    learner = QLearning(
        None, None, input_batch_data[0][:-1], input_batch_data[1][:-1], input_batch_data[2][:-1], input_batch_data[0][1:])
    learner.train_sarsa()

    pdb.set_trace()

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    simulation.lobby_holder[0].add(person1)
    simulation.lobby_holder[0].add(person2)
    simulation.lobby_holder[0].add(person3)

    batch_reward_episodes = []
    batch_policies = []
    batch_states = []
    batch_data = []
    for train_samples in range(1):
        actions_vec = []
        reward_vec = []
        state_vec = []
        state = [False, False, False, False, False, False]
        for i in range(10):
            # Dont queue past 30 iterations.
            if i < 3:
                for idx, generator in enumerate(people_gen):
                    for person in generator.sample():
                        simulation.lobby_holder[0].add(person)
            old_state = state

            # state = simulation.simple_elevator_local_state(
            #     simulation.elevators[0])
            state = simulation._next_observation()

            # state = simulation._next_observation()
            # pdb.set_trace()
            states.append(state)
            action = learner.predict_sarsa(state)
            # if action == [2]:
            #     pdb.set_trace()

            reward = simulation.step(action)
            simulation.render()
            batch_data.append(np.array([action, reward, state]))
            actions_vec.append(action)
            reward_vec.append(reward)
            found_policy.append(action)

        simulation.reset()
        batch_policies.append(actions_vec)
        batch_reward_episodes.append(reward_vec)
        batch_states.append(states)

    pdb.set_trace()
