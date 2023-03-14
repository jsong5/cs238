import numpy as np
from enum import Enum
import pdb
from copy import deepcopy

# Hard coded tactics: always have 20 floors, 4 elevators, we will test 4 policies
# Baselines: 4 way scan, and 4 way zoning scans.
# Advanced: 1 way scan and centralized 3 way RL policy
# Advanced: 1 way scan and decentralized 3 way RL policy
# Even more Advanced: Make the RL policy uncover the latent poisson rate variable and make decisions with this in a centralized 3 way controller.

"""
Elevator Actions enum
"""
ELEVATOR_UP = 1
ELEVATOR_DOWN = -1
ELEVATOR_STOP = 0
ELEVATOR_LOAD_UNLOAD = 2

"""
Num Floors
"""
NUM_FLOORS = 3
NUM_ELEVATORS = 1

# global lobby
# lobby = set()

# global done_people
# done_people = set()


class Person(object):
    """
    People object representation
    """

    def default_person_policy(self, elevator):
        direction = 0
        if elevator.previous_action == 0 or self.curr_floor == NUM_FLOORS - 1 or self.curr_floor == 0:
            return True

        if self.curr_floor <= self.target_floor:
            direction = ELEVATOR_UP
        else:
            direction = ELEVATOR_DOWN
        return elevator.previous_action == direction

    def __init__(self, id, curr_floor, target_floor, person_policy_fn=default_person_policy):
        if not (curr_floor >= 0 and target_floor < NUM_FLOORS):
            pdb.set_trace()
        assert (curr_floor >= 0 and target_floor < NUM_FLOORS)
        self.id = id
        self.target_floor = target_floor
        self.origin_floor = curr_floor
        self.curr_floor = curr_floor
        self.person_policy_fn = person_policy_fn
        self.assigned_elevator = None

    def assign_elevator(self, elevator):
        self.assigned_elevator = elevator


class Elevator(object):
    """
    Elevator object representation
    """

    def __init__(self, speed=1, highest_floor=NUM_FLOORS - 1, lowest_floor=0, capacity=20):
        # Elevator Acend speed
        self.speed = speed

        # Nuber of floors serviced
        self.num_floors = highest_floor - lowest_floor

        # Current floor
        self.curr_floor = lowest_floor

        # Bounds on the floors
        self.highest_floor = highest_floor
        self.lowest_floor = lowest_floor

        # Capacity config
        self.capacity = capacity

        # Passenger set on the elevator
        self.passengers = set()

        # Need a cool down to account for load unload.
        self.cooldown = 0

        # previous action
        self.previous_action = ELEVATOR_STOP

    # Take a simulation step from the elevator.
    def move(self, action, lobby_holder, done_set):
        total_time = 0

        if self.cooldown > 0:
            # Wait for cooldown
            self.cooldown -= 1
            return 1

        if action == ELEVATOR_LOAD_UNLOAD:
            # Load all passengers
            total_time += self.load_unload_passengers(lobby_holder, done_set)

        else:
            # Update the elevator
            new_state = self.curr_floor + action
            if new_state >= self.lowest_floor and new_state <= self.highest_floor:
                self.curr_floor = new_state
            # Update the passengers in the elevator
            for person in self.passengers:
                person.curr_floor = self.curr_floor

            self.previous_action = action
            total_time += 1

        return total_time

    # Might be sufficient.
    def load_unload_passengers(self, lobby_holder, done_set):
        new_lobby = set()
        new_cart = set()
        # Unload people
        for person in self.passengers:
            if person.target_floor == self.curr_floor:
                # Leaving, and go to done
                done_set[0].add(person)
                self.cooldown += 0.1
            else:
                # Not leaving, stay in cart
                new_cart.add(person)

        # Load people
        for person in lobby_holder[0]:
            elevator = self
            if person.person_policy_fn(person, elevator) and person.curr_floor == self.curr_floor and self.capacity > len(new_cart):
                # Leave and go to cart
                self.cooldown += 0.1
                new_cart.add(person)
            else:
                # Not leaving, stay in lobby
                new_lobby.add(person)
        self.cooldown = int(self.cooldown)
        lobby_holder[0] = new_lobby
        self.passengers = new_cart
        return 1


global num_people_generated
num_people_generated = 0


class PersonGenerator(object):
    # Generator for people on each floor.
    def __init__(self, floor, random_process=np.random.poisson, parameter_tuple=((1))):
        self.generator = random_process
        self.param = parameter_tuple
        self.floor = floor

    def make_random_person(self):
        global num_people_generated
        person = Person(num_people_generated, curr_floor=self.floor,
                        target_floor=np.random.randint(NUM_FLOORS))
        num_people_generated += 1

        if person.curr_floor == person.target_floor:
            return None
        return person

    def sample(self):
        people_list = []
        for _ in range(self.generator(self.param)):
            person = self.make_random_person()
            if person:
                people_list.append(person)
        return people_list


class Building(object):
    def __init__(self, people_gen, num_elevators):
        # Internal generator of people.
        self.floor_people = people_gen

        # Number of floors to initialize.
        self.num_floors = len(people_gen)

        # Initialize the elevators.
        self.elevators = [
            Elevator(speed=1, highest_floor=len(people_gen) - 1, lowest_floor=0) for _ in range(num_elevators)]

        # Initial time 0.
        self.time_steps = 0

        # Global sets.
        self.lobby_holder = [set()]
        self.done_people = [set()]

    def reset(self):
        for elevator in self.elevators:
            elevator.passengers = set()
            elevator.curr_floor = 0

        self.lobby_holder = [set()]
        self.done_people = [set()]

    # Takes an Action and returns next state, reward.
    def step(self, elevator_actions):
        assert (len(elevator_actions) == len(self.elevators))

        old_done_count = len(self.done_people[0])
        times = []
        for idx, elevator in enumerate(self.elevators):
            times.append(elevator.move(
                elevator_actions[idx], self.lobby_holder, self.done_people))

        people_serviced_this_round = len(self.done_people[0]) - old_done_count

        # Should always be n - 1
        return people_serviced_this_round * 10 - max(times)

    def simple_elevator_local_state(self, elevator: Elevator):
        floor = elevator.curr_floor
        extern_ups, extern_downs = direction_requests(self.lobby_holder[0])
        intern_ups, intern_downs = direction_requests(elevator.passengers)
        extern_up_up = sum(extern_ups[floor+1:])
        extern_down_down = sum(extern_downs[:floor])

        intern_up_up = sum(intern_ups)
        intern_down_down = sum(intern_downs)

        extern_up_down = sum(extern_downs[floor+1:])
        extern_down_up = sum(extern_ups[:floor])

        if (elevator.curr_floor == 0 and (intern_down_down or extern_down_up or extern_down_down)) or (elevator.curr_floor == NUM_FLOORS - 1 and (extern_up_up or intern_up_up or extern_up_down)):
            pdb.set_trace()

        return extern_up_up > 0, extern_down_down > 0, intern_up_up > 0, intern_down_down > 0, extern_up_down > 0, extern_down_up > 0

    # Displays the current state of the building
    def render(self):
        print("------------START--------------")
        print("Lobby")

        waiting_array = [0 for _ in range(NUM_FLOORS)]
        for person in self.lobby_holder[0]:
            waiting_array[person.curr_floor] += 1
        print(waiting_array)

        elevator_arrays = [
            [0 for _ in range(NUM_FLOORS)] for _ in range(NUM_ELEVATORS)]

        for idx, elevator_arr in enumerate(elevator_arrays):
            print("ELEVATOR LOCATION:", idx)
            location = np.zeros(NUM_FLOORS)
            location[self.elevators[idx].curr_floor] = 1

            print(location)

            print("INSIDE ELEVATOR:", idx)
            for person in self.elevators[idx].passengers:
                elevator_arr[person.target_floor] += 1
            print(elevator_arr)
        print("------------END--------------")


def direction_requests(people_representation: set()):
    # Up and down indicator containers for the floors.
    ups = [0 for _ in range(NUM_FLOORS)]
    downs = [0 for _ in range(NUM_FLOORS)]

    # Loop over the people and find directionality in the floors.
    for person in people_representation:
        going_up = person.target_floor - person.curr_floor
        if going_up > 0:
            ups[person.curr_floor] = 1
        elif going_up < 0:
            downs[person.curr_floor] = 1
    return ups, downs


def elevator_directions(elevator):
    # Returns indicator tuple if there exists people within the elevator wanting to go up or down
    going_up = 0
    going_down = 0
    for person in elevator.passengers:
        direction = person.target_floor - person.curr_floor
        if direction > 0:
            going_up = 1
        elif direction < 0:
            going_down = 1

    return going_down, going_up

# Of the form [lobby up, lobby down, elevator up, elevator down, ]


def scan_policy(simulation: Building):
    actions = [ELEVATOR_STOP for _ in range(NUM_ELEVATORS)]
    for idx, elevator in enumerate(simulation.elevators):
        extern_up_up, extern_down_down, intern_up_up, intern_down_down, extern_up_down, extern_down_up = simulation.simple_elevator_local_state(
            elevator)
        elevator_stall = (elevator.curr_floor == ELEVATOR_DOWN and elevator.curr_floor == 0) or (
            elevator.curr_floor == ELEVATOR_UP and elevator.curr_floor == NUM_FLOORS - 1) or (elevator.previous_action == ELEVATOR_STOP)

        # Already going up
        if elevator.previous_action == ELEVATOR_UP:
            if extern_up_up or intern_up_up or extern_up_down:
                actions[idx] = ELEVATOR_UP
            elif extern_down_down or intern_down_down or extern_down_up:
                actions[idx] = ELEVATOR_DOWN

        # Already down
        elif elevator.previous_action == ELEVATOR_DOWN:
            if extern_down_down or intern_down_down or extern_down_up:
                actions[idx] = ELEVATOR_DOWN
            elif extern_up_up or intern_up_up or extern_up_down:
                actions[idx] = ELEVATOR_UP

        # Already Stopped
        else:
            assert (elevator.previous_action == ELEVATOR_STOP)
            if extern_down_down or intern_down_down or extern_down_up:
                actions[idx] = ELEVATOR_DOWN
            elif extern_up_up or intern_up_up or extern_up_down:
                actions[idx] = ELEVATOR_UP

        # Decide to stop or not
        for person in simulation.lobby_holder[0]:
            if person.curr_floor == elevator.curr_floor:
                person_direction = (person.target_floor -
                                    elevator.curr_floor > 0)
                if person_direction == elevator.previous_action or elevator_stall:
                    actions[idx] = ELEVATOR_LOAD_UNLOAD
                    break

                if elevator.previous_action == ELEVATOR_DOWN and not (extern_down_down or intern_down_down or extern_down_up):
                    actions[idx] = ELEVATOR_LOAD_UNLOAD
                    break

                if elevator.previous_action == ELEVATOR_UP and not (extern_up_up or intern_up_up or extern_up_down):
                    actions[idx] = ELEVATOR_LOAD_UNLOAD
                    break

        for person in elevator.passengers:
            if person.target_floor == elevator.curr_floor:
                actions[idx] = ELEVATOR_LOAD_UNLOAD
                break

        if (actions[idx] == ELEVATOR_DOWN and elevator.curr_floor == 0) or (actions[idx] == ELEVATOR_UP and elevator.curr_floor == NUM_FLOORS - 1):
            pdb.set_trace()

    return actions

# Simple unit test for simulator


def unit_test_hardcoded(render=False):
    # Make sure we can initialize the elevator.
    people_gen = [PersonGenerator(floor) for floor in range(3)]
    simulation = Building(people_gen, 1)
    actions = [ELEVATOR_STOP for _ in range(1)]

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    simulation.lobby_holder[0].add(person1)
    simulation.lobby_holder[0].add(person2)
    simulation.lobby_holder[0].add(person3)

    policy = [2, 0, 1, 2, 0, 0, 1, 2, 0, -1, -1, 2]
    if render:
        simulation.render()

    for action in policy:
        simulation.step(actions)
        policy = [action]
        simulation.step(policy)
        if render:
            simulation.render()
            print(policy)
    assert (len(simulation.lobby_holder[0]) == 0 and len(
        simulation.elevators[0].passengers) == 0 and simulation.elevators[0].curr_floor == 0)
    print("HARDCODE TEST PASSED!")


# def small_state_space(simulation):


def unit_test_scan(render=False):
    people_gen = [PersonGenerator(floor) for floor in range(3)]
    simulation = Building(people_gen, 1)
    actions = [ELEVATOR_STOP for _ in range(1)]

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    simulation.lobby_holder[0].add(person1)
    simulation.lobby_holder[0].add(person2)
    simulation.lobby_holder[0].add(person3)
    if render:
        simulation.render()
    found_policy = []

    rewards = []
    for _ in range(15):
        policy = scan_policy(simulation)
        if render:
            print(policy)
        reward = simulation.step(policy)
        rewards.append(reward)
        if render:
            simulation.render()
        found_policy.append(policy)
    if render:
        print(found_policy)
        print("Reward: ", sum(rewards))
        print("Episodic Reward", rewards)
    assert (len(simulation.lobby_holder[0]) == 0 and len(
        simulation.elevators[0].passengers) == 0 and simulation.elevators[0].curr_floor == 0)
    print("SCAN TEST PASSED!")


if __name__ == "__main__":
    # Make sure we can initialize the elevator.
    # unit_test_hardcoded()
    # unit_test_scan()

    people_gen = [PersonGenerator(floor) for floor in range(NUM_FLOORS)]
    simulation = Building(people_gen, 1)
    actions = [ELEVATOR_STOP for _ in range(1)]

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    simulation.lobby_holder[0].add(person1)
    simulation.lobby_holder[0].add(person2)
    simulation.lobby_holder[0].add(person3)
    simulation.render()
    found_policy = []

    generation_rounds = [1, 3, 5]

    rewards = []
    for i in range(30):
        # Seeded generation rounds
        if i in generation_rounds:
            for idx, generator in enumerate(people_gen):
                for person in generator.sample():
                    simulation.lobby_holder[0].add(person)

        policy = scan_policy(simulation)
        print(policy)
        reward = simulation.step(policy)
        rewards.append(reward)
        simulation.render()
        found_policy.append(policy)
        print(found_policy)
        print("Reward: ", sum(rewards))
        print("Episodic Reward", rewards)
