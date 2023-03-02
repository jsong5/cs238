import numpy as np
from enum import Enum
import pdb

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
ELEVATOR_LOAD = 2
ELEVATOR_UNLOAD = 3

"""
Num Floors
"""
NUM_FLOORS = 10
NUM_ELEVATORS = 3


global waiting_area
waiting_area = [set() for _ in range(NUM_FLOORS)]


class Person(object):
    """
    People object representation
    """

    def default_person_policy(self, elevator):
        direction = 0
        if self.current_floor <= self.target_floor:
            direction = 1
        else:
            direction = -1
        return elevator.direction == direction

    def __init__(self, id, target_floor, current_floor, elevator_idx=None, person_policy_fn=default_person_policy):
        self.id = id
        self.target_floor = target_floor
        self.elevator_idx = elevator_idx
        self.assigned_elevator_idx = elevator_idx
        self.current_floor = current_floor
        self.person_policy_fn = person_policy_fn
        self.going_up = (self.current_floor <= self.target_floor)


class Elevator(object):
    """
    Elevator object representation
    """

    def __init__(self, speed=1, highest_floor=10, lowest_floor=0, capacity=20):
        # Initialization work
        self.speed = speed
        self.num_floors = highest_floor - lowest_floor
        self.floor = lowest_floor
        self.highest_floor = highest_floor
        self.lowest_floor = lowest_floor
        self.capacity = capacity
        self.direction = ELEVATOR_STOP
        self.elevator_cool_down = 0
        self.passengers = set()

    # Take a simulation step from the elevator.
    def move(self, action):
        total_time = 0
        if self.elevator_cool_down != 0:
            self.elevator_cool_down -= 1
            return 1
        if action == ELEVATOR_LOAD:
            total_time += self.load_all_passengers()
        elif action == ELEVATOR_UNLOAD:
            total_time += self.unload_passengers()
        else:
            # Stay Moving
            new_state = self.floor + action
            if new_state >= self.lowest_floor and new_state <= self.highest_floor:
                self.floor = new_state
            total_time += 1

            if action == ELEVATOR_DOWN:
                self.direction = ELEVATOR_DOWN
            elif action == ELEVATOR_UP:
                self.direction = ELEVATOR_UP

        return total_time

    # Might be sufficient.
    def load_all_passengers(self):
        for person in waiting_area[self.floor]:
            if person.person_policy_fn(self):
                self.passengers.add(person)
                waiting_area[self.floor].remove(person)
        return 1

    #  Takes a target floor and unloads all passengers.
    def unload_passengers(self):
        outbound = set()
        for passenger in self.passengers:
            if passenger.target_floor == self.floor:
                outbound.add(passenger)
        for passenger in outbound:
            self.passengers.remove(passenger)
            waiting_area[self.floor].add(passenger)
        return 1


class PersonGenerator(object):
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

        # Metadata for tracking.
        self.people_to_elevator = dict()
        self.evelator_to_people = dict()

    # Takes an Action and returns next state, reward.
    def step(self, elevator_actions):
        assert (len(elevator_actions) == len(self.elevators))
        times = []
        for idx, elevator in enumerate(self.elevators):
            times.append(elevator.move(elevator_actions[idx]))
        print(times)


def simple_state(simulation):
    # returns a up and down array with indicatiors that people are interested in a particular direction.
    ups = [0 for _ in range(NUM_FLOORS)]
    downs = [0 for _ in range(NUM_FLOORS)]
    for idx, lobby in enumerate(waiting_area):
        for person in lobby:
            if person.going_up:
                ups[idx] = 1
            else:
                downs[idx] = 1
        if downs[idx] == 1 and ups[idx] == 1:
            break
    return ups, downs


def scan_policy(simulation):
    ups, downs = simple_state(simulation)
    actions = [ELEVATOR_STOP for _ in range(NUM_ELEVATORS)]

    for elevator_idx in range(NUM_ELEVATORS):
        elevator = simulation.elevators[elevator_idx]
        if elevator.floor == NUM_FLOORS:
            actions[elevator_idx] = ELEVATOR_DOWN
        elif elevator.floor == 0:
            actions[elevator_idx] = ELEVATOR_UP
        else:
            if elevator.direction == ELEVATOR_UP:
                if np.sum(ups[elevator.floor:]) > 0:
                    actions[elevator_idx] = ELEVATOR_UP
            else:
                if np.sum(downs[:elevator.floor]) > 0:
                    actions[elevator_idx] = ELEVATOR_DOWN
    return actions


if __name__ == "__main__":
    people = [PersonGenerator(NUM_ELEVATORS) for _ in range(10)]
    simulation = Building(people, NUM_ELEVATORS)
    iters = 0
    actions = [ELEVATOR_STOP for _ in range(NUM_ELEVATORS)]
    person = Person(0, 10, 0)

    waiting_area[5].add(person)

    pdb.set_trace()
    for _ in range(10):
        simulation.step(actions)
        policy = scan_policy(simulation)
        simulation.step(policy)
    pdb.set_trace()
