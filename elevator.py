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

global lobby
lobby = set()

global done_people
done_people = set()


class Person(object):
    """
    People object representation
    """

    def default_person_policy(self, elevator):
        direction = 0
        if elevator.previous_action == 0:
            return True

        if self.curr_floor <= self.target_floor:
            direction = ELEVATOR_UP
        else:
            direction = ELEVATOR_DOWN
        return elevator.previous_action == direction

    def __init__(self, id, curr_floor, target_floor, person_policy_fn=default_person_policy):
        assert (curr_floor >= 0 and target_floor < NUM_FLOORS)
        self.id = id
        self.target_floor = target_floor
        self.origin_floor = curr_floor
        self.curr_floor = curr_floor
        self.person_policy_fn = person_policy_fn
        self.assigned_elevator = None

    def assign_elevator(self, elevator):
        self.assigned_elevator = elevator


lobby = set()
done_people = set()


class Elevator(object):
    """
    Elevator object representation
    """

    def __init__(self, speed=1, highest_floor=10, lowest_floor=0, capacity=20):
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
    def move(self, action):
        total_time = 0

        if self.cooldown > 0:
            # Wait for cooldown
            self.cooldown -= 1
            return 1

        if action == ELEVATOR_LOAD_UNLOAD:
            # Load all passengers
            total_time += self.load_unload_passengers()

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
    def load_unload_passengers(self):
        global lobby
        global done_people

        new_lobby = set()
        new_cart = set()
        # Unload people
        for person in self.passengers:
            if person.target_floor == self.curr_floor:
                # Leaving, and go to done
                done_people.add(person)
                self.cooldown += 1
            else:
                # Not leaving, stay in cart
                new_cart.add(person)

        # Load people
        for person in lobby:
            elevator = self
            if person.person_policy_fn(person, elevator) and person.curr_floor == self.curr_floor:
                # Leave and go to cart
                self.cooldown += 1
                new_cart.add(person)
            else:
                # Not leaving, stay in lobby
                new_lobby.add(person)
        lobby = new_lobby
        self.passengers = new_cart

        return 1


class PersonGenerator(object):
    # Generator for people on each floor.
    def __init__(self, random_process=np.random.poisson, parameter_tuple=(1)):
        self.generator = random_process
        self.param = parameter_tuple

    def sample(self):
        return self.generator(*self.param)


class Building(object):
    def __init__(self, people_gen, num_elevators):
        # Internal generator of people.
        self.floor_people = people_gen

        # Number of floors to initialize.
        self.num_floors = len(people_gen)

        # Initialize the elevators
        self.elevators = [
            Elevator(speed=1, highest_floor=len(people_gen) - 1, lowest_floor=0) for _ in range(num_elevators)]

        # Initial time 0
        self.time_steps = 0

    # Takes an Action and returns next state, reward.
    def step(self, elevator_actions):
        assert (len(elevator_actions) == len(self.elevators))

        old_done_count = len(done_people)
        times = []
        for idx, elevator in enumerate(self.elevators):
            times.append(elevator.move(elevator_actions[idx]))

        people_serviced_this_round = len(done_people) - old_done_count

        # Should always be n - 1
        return people_serviced_this_round * 10 - max(times)

    # Displays the current state of the building
    def render(self):
        print("------------START--------------")
        print("Lobby")

        waiting_array = [0 for _ in range(NUM_FLOORS)]
        for person in lobby:
            waiting_array[person.curr_floor] += 1
        print(waiting_array)

        elevator_arrays = [
            [0 for _ in range(NUM_FLOORS)] for _ in range(NUM_ELEVATORS)]

        for idx, elevator_arr in enumerate(elevator_arrays):
            print("ELEVATOR LOCATION:", idx)
            location = np.zeros(NUM_FLOORS)
            location[self.elevators[idx].curr_floor] = 1

            print(location)

            print("ELEVATOR TARGETS:", idx)
            for person in self.elevators[idx].passengers:
                elevator_arr[person.target_floor] += 1
            print(elevator_arr)
        print("------------END--------------")


def lobby_direction_requests(people_representation: set()):
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


def scan_policy(simulation: Building):
    ups, downs = lobby_direction_requests(lobby)
    actions = [ELEVATOR_STOP for _ in range(NUM_ELEVATORS)]
    for idx, elevator in enumerate(simulation.elevators):
        old_direction = elevator.previous_action
        #  (down indicator, up indicator)

        # Check load from lobby
        for person in lobby:
            person_direction = 0
            if person.target_floor - person.origin_floor > 0:
                person_direction = ELEVATOR_UP
            elif person.target_floor - person.origin_floor < 0:
                person_direction = ELEVATOR_DOWN

            if person.origin_floor == elevator.curr_floor and (person_direction == elevator.previous_action or elevator.previous_action == ELEVATOR_STOP):
                actions[idx] = ELEVATOR_LOAD_UNLOAD
                break

        # Check unload from cart
        for person in elevator.passengers:
            if person.target_floor == elevator.curr_floor:
                actions[idx] = ELEVATOR_LOAD_UNLOAD
                break

            # break if we need to stop
        if actions[idx] == ELEVATOR_LOAD_UNLOAD:
            continue

        # Otherwise we check if we need to keep moving.

        #  (down indicator, up indicator)
        inter_directions = elevator_directions(elevator)
        #  (down indicator, up indicator)
        extern_directions = (
            sum(downs[:elevator.curr_floor]), sum(ups[elevator.curr_floor:]))

        if old_direction == ELEVATOR_UP or old_direction == ELEVATOR_STOP:
            if inter_directions[1] == 1 or extern_directions[1] > 0:
                actions[idx] = ELEVATOR_UP
            elif inter_directions[0] == 1 or extern_directions[0] > 0:
                actions[idx] = ELEVATOR_DOWN

        if old_direction == ELEVATOR_DOWN or old_direction == ELEVATOR_STOP:
            if inter_directions[0] == 1 or extern_directions[0] > 0:
                actions[idx] = ELEVATOR_DOWN
            elif inter_directions[1] == 1 or extern_directions[1] > 0:
                actions[idx] = ELEVATOR_UP
    return actions

# Simple unit test for simulator


def unit_test_hardcoded(render=False):
    # Make sure we can initialize the elevator.
    people_gen = [PersonGenerator(1) for _ in range(3)]
    simulation = Building(people_gen, 1)
    actions = [ELEVATOR_STOP for _ in range(1)]

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    lobby.add(person1)
    lobby.add(person2)
    lobby.add(person3)

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
    assert (len(lobby) == 0 and len(
        simulation.elevators[0].passengers) == 0 and simulation.elevators[0].curr_floor == 0)
    print("TEST PASSED!")


if __name__ == "__main__":
    # Make sure we can initialize the elevator.
    people_gen = [PersonGenerator(1) for _ in range(3)]
    simulation = Building(people_gen, 1)
    actions = [ELEVATOR_STOP for _ in range(1)]

    person1 = Person(0, 0, 2)
    person2 = Person(1, 1, 2)
    person3 = Person(2, 1, 0)

    lobby.add(person1)
    lobby.add(person2)
    lobby.add(person3)

    simulation.render()
    found_policy = []

    rewards = []
    for _ in range(15):
        policy = scan_policy(simulation)
        print(policy)
        reward = simulation.step(policy)
        rewards.append(reward)
        simulation.render()
        found_policy.append(policy)
    print(found_policy)
    print("Reward: ", sum(rewards))
    print("Episodic Reward", rewards)
