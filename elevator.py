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
ELEVATOR_LOAD_UNLOAD = 2

"""
Num Floors
"""
NUM_FLOORS = 10
NUM_ELEVATORS = 1


global lobby


class Person(object):
    """
    People object representation
    """

    def default_person_policy(self, elevator):
        direction = 0
        if self.current_floor <= self.target_floor:
            direction = ELEVATOR_UP
        else:
            direction = ELEVATOR_DOWN
        return elevator.previous_action == direction

    def __init__(self, id, target_floor, current_floor, person_policy_fn=default_person_policy):
        self.id = id
        self.target_floor = target_floor
        self.origin_floor = current_floor
        self.curr_floor = current_floor
        self.person_policy_fn = person_policy_fn
        self.assigned_elevator = None

    def assign_elevator(self, elevator):
        self.assigned_elevator = elevator


lobby = set()


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
        self.current_floor = lowest_floor

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
            new_state = self.current_floor + action
            if new_state >= self.lowest_floor and new_state <= self.highest_floor:
                self.current_floor = new_state
            # Update the passengers in the elevator
            for person in self.passengers:
                person.curr_floor = self.current_floor

            self.previous_action = action

            total_time += 1

        return total_time

    # Might be sufficient.
    def load_unload_passengers(self):
        # Unload and poof
        for person in self.elevator.passengers:
            if person.target_floor == self.elevator.current_floor:
                self.elevator.remove(person)

        # Load people
        for person in lobby:
            if person.person_policy_fn(self):
                self.passengers.add(person)
                lobby.remove(person)
                self.cooldown += 1

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
        times = []
        for idx, elevator in enumerate(self.elevators):
            times.append(elevator.move(elevator_actions[idx]))
        print(times)


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

            if person.origin_floor == elevator.current_floor and person_direction == elevator.previous_action:
                actions[idx] = ELEVATOR_LOAD_UNLOAD
                break

        # Check unload from cart
        for person in elevator.passengers:
            if person.target_floor == elevator.current_floor:
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
            sum(downs[:elevator.current_floor]), sum(ups[elevator.current_floor:]))
        if old_direction == ELEVATOR_UP or old_direction == ELEVATOR_STOP:
            if inter_directions[1] == 1 or extern_directions[1]:
                actions[idx] = ELEVATOR_UP
        if old_direction == ELEVATOR_DOWN or old_direction == ELEVATOR_STOP:
            if inter_directions[0] == 1 or extern_directions[0]:
                actions[idx] = ELEVATOR_DOWN

    return actions


if __name__ == "__main__":
    people = [PersonGenerator(NUM_ELEVATORS) for _ in range(10)]
    simulation = Building(people, NUM_ELEVATORS)
    iters = 0
    actions = [ELEVATOR_STOP for _ in range(NUM_ELEVATORS)]
    person1 = Person(0, 9, 5)
    person2 = Person(0, 9, 3)
    person3 = Person(0, 2, 9)

    lobby.add(person1)
    lobby.add(person2)
    lobby.add(person3)

    # pdb.set_trace()
    for _ in range(10):
        simulation.step(actions)
        policy = scan_policy(simulation)
        simulation.step(policy)
        print(policy)
    # pdb.set_trace()
