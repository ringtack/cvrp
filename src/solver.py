from vrp_parser import VRPInstance
import numpy as np
import math
import copy as cp
from collections import defaultdict
import matplotlib.pyplot as plt


from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import NoImprovement, MaxRuntime
from alns.select import RouletteWheel
from alns.select import MABSelector
from mabwiser.mab import MAB, LearningPolicy

# from alns.Statistics import Statistics


class VRPState:
    """
    State for the problem.
    - number of vehicles used
    - mapping a vehicle to a route
    """

    def __init__(
        self,
        vrp_instance: VRPInstance,
        vehicles_to_route,
        num_vehicles,
        vehicle_to_capacity,
    ):
        self.vrp_instance = vrp_instance
        self.max_num_vehicles = self.vrp_instance.num_vehicles
        self.num_customers = self.vrp_instance.num_customers
        self.vehicle_capacity = self.vrp_instance.vehicle_capacity
        self.customer_demand = self.vrp_instance.demandOfCustomer
        self.customer_x = self.vrp_instance.xCoordOfCustomer
        self.customer_y = self.vrp_instance.yCoordOfCustomer

        # what makes a solution different
        self.num_vehicles = num_vehicles
        self.vehicle_to_route = vehicles_to_route
        self.vehicle_to_capacity = vehicle_to_capacity
        self.unassigned_customers = (
            []
        )  # this is used for global destroy and repair operators
        self.removed_customers = (
            {}
        )  # used for local operatores, where you still want to keep them in the same route
        self.customers_distances = None
        self.customers_indices = None

    def copy(self):
        pass

    def objective(self):
        # computing the distance calculated by all vehicles
        cost = 0
        distance = 0
        # completely ignore anything that goes past max_num_vehicles
        if self.num_vehicles > self.max_num_vehicles + 1:
            return float("inf")

        depot_x = self.customer_x[0]
        depot_y = self.customer_y[0]

        unserved_customers = set()
        for i in range(1, self.num_customers):
            unserved_customers.add(i)
        for i in range(self.num_vehicles):
            customer_list = self.vehicle_to_route[i]
            capacity_served = 0
            # for each route calculate the distance traveled
            for j in range(len(customer_list)):
                curr_customer = customer_list[j]
                unserved_customers.remove(curr_customer)
                capacity_served += self.customer_demand[curr_customer]
                if j == 0:
                    distance += math.sqrt(
                        (self.customer_x[curr_customer] - depot_x) ** 2
                        + (self.customer_y[curr_customer] - depot_y) ** 2
                    )
                else:
                    # now that we are at this client, how costly was it to get here
                    prev_customer = customer_list[j - 1]
                    distance += math.sqrt(
                        (
                            self.customer_x[curr_customer]
                            - self.customer_x[prev_customer]
                        )
                        ** 2
                        + (
                            self.customer_y[curr_customer]
                            - self.customer_y[prev_customer]
                        )
                        ** 2
                    )

            # getting back to the original point (0,0)
            if len(customer_list) > 0:
                last_customer = customer_list[-1]
                distance += math.sqrt(
                    (self.customer_x[last_customer] - depot_x) ** 2
                    + (self.customer_y[last_customer] - depot_y) ** 2
                )  # getting back to the lot
            # if we exceed capaity at any point also ignore it
            if capacity_served > self.vehicle_capacity:
                return float("inf")
        # cost += len(self.vehicle_to_route[self.num_vehicles]) * 100000
        unserved_demand = 0
        _, unserved = list(self.vehicle_to_route.items())[-1]
        for c in unserved:
            unserved_demand += self.customer_demand[c]
        cost += unserved_demand * 10000
        cost += distance

        # we don't inmediately return infinite here since we believe that routes with fewer customers in the fake vehicle are closer to the true solution
        # print(cost)
        return cost

    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for idx, route in self.vehicle_to_route.items():
            if customer in route:
                return idx, route

        raise ValueError(f"Solution does not contain customer {customer}.")


def get_distance_between(state, c1, c2):
    return math.sqrt(
        (state.customer_x[c1] - state.customer_x[c2]) ** 2
        + (state.customer_y[c1] - state.customer_y[c2]) ** 2
    )  # getting back to the lot


def two_opt_destroy(
    state: VRPState, rnd_state, **kwargs
):  # this switch is within one route
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    # skip the last route which is the ghost one
    route_idx = rnd_state.choice(len(state.vehicle_to_route) - 1, 1, replace=False)[0]
    route = state.vehicle_to_route[route_idx]
    if len(route) <= 3:
        return cp_state
    length = rnd_state.choice(np.arange(0, len(route)), 1, replace=False)[0]
    start = rnd_state.choice(np.arange(0, len(route) - length), 1, replace=False)[0]
    end = start + length
    cp_state.removed_customers[route_idx] = [start, end]
    return cp_state


# this is basically changing two edges by taking some middle chunk and flipping it so that the conenctin parts are diff
def two_opt_repair(
    state: VRPState, rnd_state, **kwargs
):  # this is like best local repair
    cp_state = cp.deepcopy(state)
    to_insert = cp.deepcopy(cp_state.removed_customers)
    cp_state.removed_customers.clear()
    cp_state.unassigned_customers.clear()
    for car, customers in to_insert.items():  # this loop basically only happens once
        start = customers[0]
        end = customers[1]
        cp_state.vehicle_to_route[car][start:end] = state.vehicle_to_route[car][
            start:end
        ][::-1]
    return cp_state


def route_two_opt_destroy(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    route_indices = rnd_state.choice(len(state.vehicle_to_route) - 1, 2, replace=False)
    r1 = cp_state.vehicle_to_route[route_indices[0]]
    r2 = cp_state.vehicle_to_route[route_indices[1]]
    if len(r1) < 2 or len(r2) < 2:
        return cp_state
    # subtract 1 because we can get the end of the list
    r1_edge_start = rnd_state.choice(len(r1) - 1, 1, replace=False)[0]
    r2_edge_start = rnd_state.choice(len(r2) - 1, 1, replace=False)[0]
    cp_state.removed_customers[route_indices[0]] = r2[r2_edge_start + 1 :]
    cp_state.removed_customers[route_indices[1]] = r1[r1_edge_start + 1 :]
    cp_state.vehicle_to_capacity[route_indices[0]] += find_demand(
        state, r1[r1_edge_start + 1 :]
    )
    cp_state.vehicle_to_capacity[route_indices[1]] += find_demand(
        state, r2[r2_edge_start + 1 :]
    )
    cp_state.vehicle_to_route[route_indices[1]] = r2[: r2_edge_start + 1]
    cp_state.vehicle_to_route[route_indices[0]] = r1[: r1_edge_start + 1]
    return cp_state


def route_two_opt_repair(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    for r, value in state.removed_customers.items():
        cp_state.vehicle_to_route[r] = cp_state.vehicle_to_route[r] + value
        cp_state.vehicle_to_capacity[r] -= find_demand(state, value)
    return cp_state


# this is for blocks of customers (a segment) to a close neighboring cluster
def switch_across_routes(
    state: VRPState, rnd_state, **kwargs
):  # switches segments across routes
    cp_state = cp.deepcopy(state)
    route_idx = list(
        rnd_state.choice(len(state.vehicle_to_route) - 1, 1, replace=False)
    )[0]
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    route = state.vehicle_to_route[route_idx]
    if len(route) < 2:
        return cp_state
    to_swap = rnd_state.choice(route, 1, replace=False)[0]
    to_swap_idx = route.index(to_swap)
    length = min(2, len(route) - to_swap_idx)
    # length = rnd_state.choice(len(route) - to_swap_idx, 1, replace=False)[0]

    idx = 0
    closest = state.customers_indices[to_swap][idx]
    while idx < state.num_customers and (
        closest in state.vehicle_to_route[state.num_vehicles]
        or cp_state.find_route(closest)[0] == route_idx
    ):
        idx += 1
        closest = state.customers_indices[to_swap][idx]

    if idx == state.num_customers - 1:
        print("ran out of customers")
        return

    alt_route_idx, alt_route = state.find_route(closest)
    closest_idx = alt_route.index(closest)
    # alt_length = rnd_state.choice(len(alt_route) - closest_idx, 1, replace=False)[0]
    alt_length = min(2, len(alt_route) - closest_idx)
    # segment_length = min(len(alt_route) - closest_idx, length)
    cp_state.removed_customers[route_idx] = (
        to_swap_idx,
        alt_route[closest_idx : closest_idx + alt_length],
    )
    cp_state.removed_customers[alt_route_idx] = (
        closest_idx,
        route[to_swap_idx : to_swap_idx + length],
    )

    # print("switching")
    # print(length)
    # print(find_demand(state, route[to_swap_idx:to_swap_idx + length]))
    # print(alt_length)
    # print(find_demand(state, alt_route[closest_idx:closest_idx + alt_length]))
    if route_idx < state.num_vehicles:
        cp_state.vehicle_to_capacity[route_idx] += find_demand(
            state, route[to_swap_idx : to_swap_idx + length]
        )
    if alt_route_idx < state.num_vehicles:
        cp_state.vehicle_to_capacity[alt_route_idx] += find_demand(
            state, alt_route[closest_idx : closest_idx + alt_length]
        )

    cp_state.vehicle_to_route[route_idx] = (
        route[:to_swap_idx] + route[to_swap_idx + length :]
    )
    cp_state.vehicle_to_route[alt_route_idx] = (
        alt_route[:closest_idx] + alt_route[closest_idx + alt_length :]
    )
    return cp_state


# do best insertion
def best_leftover_insertion(state: VRPState):
    if state.objective() == float("inf"):
        return

    temp_cp = cp.deepcopy(state.vehicle_to_route[state.num_vehicles])
    for c in temp_cp:
        best_car = None
        best_pos = None
        lowest_cost = float("inf")
        for car, cap in state.vehicle_to_capacity.items():
            if car == state.num_vehicles:
                continue
            if state.customer_demand[c] <= cap:
                pos, cost = find_best_pos(state.vehicle_to_route[car], c, state)
                if cost < lowest_cost:
                    best_car = car
                    best_pos = pos
                    lowest_cost = cost
        if best_car is not None:
            print("ridding")
            print(c)
            print(state.customer_demand[c])
            print(state.vehicle_to_capacity[best_car])
            # print(find_demand(state, state.vehicle_to_route[best_car]) == state.vehicle_capacity - state.vehicle_to_capacity[best_car])
            state.vehicle_to_route[best_car].insert(best_pos, c)
            state.vehicle_to_capacity[best_car] -= state.customer_demand[c]
            state.vehicle_to_route[state.num_vehicles].remove(c)
            print(state.objective())


def find_best_pos(route, customer, state):
    lowest_dist = float("inf")
    best_pos = None
    if len(route) == 0:
        return 0, get_distance_between(state, customer, 0)
    for i in range(len(route)):
        c = route[i]
        cost = 0
        if i == 0 or i == len(route) - 1:
            cost = get_distance_between(state, customer, 0)
            cost += get_distance_between(state, customer, c)
        else:
            cost = get_distance_between(state, customer, c)
            cost += get_distance_between(state, customer, i + 1)
        if cost < lowest_dist:
            lowest_dist = cost
            best_pos = i + 1
            if i == 0:
                best_pos = 0
            elif i == len(route) - 1:
                best_pos = len(route)
    return best_pos, lowest_dist


def insert_across_routes(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    segments = cp.deepcopy(state.removed_customers)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    for veh, pair in segments.items():
        route = state.vehicle_to_route[veh]
        cp_state.vehicle_to_route[veh] = route[: pair[0]] + pair[1] + route[pair[0] :]
        if veh < state.num_vehicles:
            cp_state.vehicle_to_capacity[veh] -= find_demand(state, pair[1])

    best_leftover_insertion(cp_state)
    return cp_state


def find_demand(state: VRPState, route):
    d = 0
    for c in route:
        d += state.customer_demand[c]
    return d


def random_removal(state: VRPState, rnd_state, n_remove=1):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    # remove a random customer from the currently functional route
    route_idx = rnd_state.choice(len(state.vehicle_to_route) - 1, 1, replace=False)[0]
    route = cp_state.vehicle_to_route[route_idx]
    if len(route) == 0:
        return cp_state
    if len(route) < 3:
        n_remove = 1
    else:
        n_remove = 2
    to_remove = rnd_state.choice(len(route), n_remove, replace=False)
    to_remove_customers = []
    for el in to_remove:
        to_remove_customers.append(route[el])
    for removed_customer in to_remove_customers:
        cp_state.unassigned_customers.append(removed_customer)
        route.remove(removed_customer)
        # update capacity for the vehicles whose customers have been removed
        cp_state.vehicle_to_capacity[route_idx] += cp_state.customer_demand[
            removed_customer
        ]
    return cp_state


def best_global_repair(state: VRPState, rnd_state, **kwargs):
    # check if there is a better place to insert the customer in
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.vehicle_to_route[state.num_vehicles] = []
    total_unassigned = set()
    for c in state.unassigned_customers:
        total_unassigned.add(c)
    for c in state.vehicle_to_route[state.num_vehicles]:
        total_unassigned.add(c)
    for unassigned in total_unassigned:
        depot_idx = 0
        best_cost, best_car, best_pos = float("inf"), None, None
        costs = []
        cars = []
        positions = []
        for car, route in state.vehicle_to_route.items():
            if car == state.num_vehicles:
                continue
            # if this route can service the customer
            if (
                cp_state.vehicle_to_capacity[car]
                >= cp_state.customer_demand[unassigned]
            ):
                # cost to insert a customer at a given point in the program
                cost = None
                if (len(route)) == 0:
                    cost = get_distance_between(state, unassigned, depot_idx) * 2
                    if cost < best_cost:
                        best_cost = cost
                        best_car = car
                        best_pos = 0
                else:
                    for i in range(len(route)):
                        cost = get_distance_between(state, unassigned, route[i])
                        if i == 0 or i == len(route) - 1:
                            # now getting to the new first or getting back to the initial from first
                            cost += get_distance_between(state, unassigned, depot_idx)
                        else:
                            # from the new one back to the +1 that it was connected to
                            cost += get_distance_between(
                                state, unassigned, route[i + 1]
                            )
                        if cost < best_cost:
                            best_cost = cost
                            best_car = car
                            best_pos = i + 1
                            if i == 0:
                                best_pos = 0
                            elif i == len(route) - 1:
                                best_pos = len(route)
                costs.append(cost)
                cars.append(car)
                positions.append(best_pos)
        # it might be the case that the nodes we removed cannot be properly reinserted into the routes given the order in which we go through them, in that case we don't want to consider this a good solution
        if best_pos == None:
            cp_state.vehicle_to_route[state.num_vehicles].append(unassigned)
        else:
            if len(cars) > 3:
                # don't always choose the best solution
                costs = np.array(costs)
                cars = np.array(cars)
                positions = np.array(positions)
                sorted_indices = np.argsort(costs)
                cars_sorted = cars[sorted_indices]
                positions_sorted = positions[sorted_indices]
                best_car_idx = rnd_state.choice(len(cars) // 2, 1, replace=False)[0]
                best_car = cars_sorted[best_car_idx]
                best_pos = positions_sorted[best_car_idx]
            cp_state.vehicle_to_route[best_car].insert(
                best_pos, unassigned
            )  # python insert displaces the item in that current position and pushes everything back
            cp_state.vehicle_to_capacity[best_car] -= cp_state.customer_demand[
                unassigned
            ]
    return cp_state


def random_fake_veh_insertion(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.fake_vehicle_customers.clear()
    total_unassigned = set()
    for c in state.unassigned_customers:
        total_unassigned.add(c)
    for c in state.fake_vehicle_customers:
        total_unassigned.add(c)
    for c in total_unassigned:
        r = rnd_state.choice(len(cp_state.vehicle_to_route), 1, replace=False)[0]
        if cp_state.vehicle_to_capacity[r] < cp_state.customer_demand[c]:
            continue
        route = cp_state.vehicle_to_route[r]
        if len(route) == 0:
            cp_state.fake_vehicle_customers.add(c)
            continue
        insertion_pos = rnd_state.choice(len(route), 1, replace=False)[0]
        cp_state.vehicle_to_route[r].insert(insertion_pos, c)
    return cp_state


# greedily repairs an entire route
def reorder_one(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    # for car, route in state.vehicle_to_route.items():
    car = rnd_state.choice(len(state.vehicle_to_route) - 1, 1, replace=False)[0]
    route = state.vehicle_to_route[car]
    cp_state.vehicle_to_route[car] = set(route)

    return cp_state


def greedy_repair(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)

    def find_closest_customer(veh_idx, customers):
        closest_customer = None
        closest_dist = float("inf")
        for c in customers:
            dist = get_distance_between(state, c, veh_idx)
            if dist < closest_dist:
                closest_dist = dist
                closest_customer = c
        return closest_customer

    for car, route in state.vehicle_to_route.items():  # this would only happen once
        if type(route) is not set:
            continue
        unserved_customers = route
        new_ordering = []
        veh_idx = 0
        while len(unserved_customers) > 0:
            new_customer = find_closest_customer(veh_idx, unserved_customers)
            unserved_customers.remove(new_customer)
            new_ordering.append(new_customer)
            veh_idx = new_customer
        cp_state.vehicle_to_route[car] = new_ordering
    return cp_state


# switches two customers
def relocate_customer_destroy(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    routes = rnd_state.choice(len(state.vehicle_to_route), 2, replace=False)
    # routes = np.append(routes,(len(state.vehicle_to_route)-1))
    r1 = cp_state.vehicle_to_route[routes[0]]
    r2 = cp_state.vehicle_to_route[routes[1]]
    if len(r1) == 0 or len(r2) == 0:
        return cp_state
    customer_r1 = rnd_state.choice(r1, 1, replace=False)[0]
    customer_r2 = rnd_state.choice(r2, 1, replace=False)[0]

    r1_insertion, _ = find_best_pos(r1, customer_r2, state)
    r2_insertion, _ = find_best_pos(r2, customer_r1, state)

    cp_state.removed_customers[routes[0]] = (routes[1], customer_r1, r1_insertion)
    cp_state.removed_customers[routes[1]] = (routes[0], customer_r2, r2_insertion)
    r1.remove(customer_r1)
    r2.remove(customer_r2)
    if routes[0] < state.num_vehicles:
        cp_state.vehicle_to_capacity[routes[0]] += state.customer_demand[customer_r1]
    if routes[1] < state.num_vehicles:
        cp_state.vehicle_to_capacity[routes[1]] += state.customer_demand[customer_r2]
    return cp_state


def relocate_customer_repair(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.removed_customers.clear()
    for orig_route, new in state.removed_customers.items():
        cp_state.vehicle_to_route[new[0]].insert(new[2], new[1])
        if new[0] < state.num_vehicles:
            cp_state.vehicle_to_capacity[new[0]] -= state.customer_demand[new[1]]

    best_leftover_insertion(cp_state)
    return cp_state


# switches with cloest neighbor
def relocate_neighbor_one(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    # this route can be the one that does not exist
    route_idx = rnd_state.choice(len(state.vehicle_to_route) - 1, 1, replace=False)[0]
    # route_idx = len(state.vehicle_to_route)-1
    route = cp_state.vehicle_to_route[route_idx]
    if len(route) == 0:
        return cp_state
    to_swap = rnd_state.choice(route, 1, replace=False)[0]
    to_swap_idx = route.index(to_swap)

    idx = 0
    closest = state.customers_indices[to_swap][idx]
    while idx < state.num_customers and (
        closest in cp_state.vehicle_to_route[state.num_vehicles]
        or cp_state.find_route(closest)[0] == route_idx
    ):
        idx += 1
        closest = state.customers_indices[to_swap][idx]
    if idx == state.num_customers:
        return cp_state
    other_route_idx, other_route = cp_state.find_route(closest)
    closest_idx = other_route.index(closest)
    other_route.remove(closest)
    route.remove(to_swap)

    # find the best place to insert them (greedy insertion)
    # other_insertion, _ = find_best_pos(other_route, to_swap, cp_state)
    # to_swap_insertion, _ = find_best_pos(route, closest, cp_state)

    # this means we will be inserting into other_route, the to_swap customer in the best place
    cp_state.removed_customers[route_idx] = (other_route_idx, to_swap, closest_idx)
    if route_idx < state.num_vehicles:
        cp_state.vehicle_to_capacity[route_idx] += state.customer_demand[to_swap]
    cp_state.removed_customers[other_route_idx,] = (route_idx, closest, to_swap_idx)
    if other_route_idx < state.num_vehicles:
        cp_state.vehicle_to_capacity[other_route_idx] += state.customer_demand[closest]
    return cp_state


def construct_distances_bw_customers(state: VRPState):
    ans = []
    indices = []
    for i in range(0, state.num_customers):
        dists = []
        for j in range(0, state.num_customers):
            if i == 0 or j == 0:
                dists.append(float("inf"))
            elif i == j:
                dists.append(float("inf"))
            else:
                dists.append(get_distance_between(state, i, j))
        indices.append(np.argsort(dists))
        dists.sort()
        ans.append(dists)
    return ans, indices


def begin_search(vrp_instance):
    seed = np.random.randint(1, 1000000)

    # THESE ARE PARAMETERS WE CAN PROBABLY PLAY WITH
    # acceptance thingy, as we go down we only allow a lower "worse" than current solution
    epsilons = [0.2, 0.1, 0.05]
    accept_probs = [
        0.5,
        0.25,
        0.10,
    ]  # probability we accept a solution at most epsilon percentage worse than current best
    initial_veh_to_customer, initial_num_vehicles, vehicle_to_capacity, unassigned = (
        vrp_instance.construct_intial_solution()
    )
    initial_state = VRPState(
        vrp_instance, initial_veh_to_customer, initial_num_vehicles, vehicle_to_capacity
    )
    # the initial solution might not have used up all cars
    initial_state.num_vehicles = 0
    dists, indices = construct_distances_bw_customers(initial_state)
    initial_state.customers_distances = dists
    initial_state.customers_indices = indices

    for v, r in initial_veh_to_customer.items():
        if len(r) > 0:
            initial_state.num_vehicles += 1
    initial_state.vehicle_to_route[initial_state.num_vehicles] = list(unassigned)
    curr_state = initial_state
    # destroy_operators = [random_removal, two_opt_destroy, switch_across_routes, reorder_one, relocate_customer_destroy, relocate_neighbor_one]
    # repair_operators = [best_global_repair, two_opt_repair, insert_across_routes, greedy_repair, relocate_customer_repair]
    destroy_operators = [
        random_removal,
        route_two_opt_destroy,
        two_opt_destroy,
        switch_across_routes,
        relocate_customer_destroy,
        relocate_neighbor_one,
    ]
    repair_operators = [
        best_global_repair,
        route_two_opt_repair,
        two_opt_repair,
        insert_across_routes,
        relocate_customer_repair,
    ]

    for i in range(len(epsilons)):
        print(curr_state.objective())
        epsilon = epsilons[i]
        accept_prob = accept_probs[i]
        alns = ALNS(np.random.RandomState(seed))
        # destroy_operators = destroy_list[i]
        # repair_operators = repair_list[i]

        destroy_num = len(destroy_operators)
        for d in destroy_operators:
            alns.add_destroy_operator(d)

        repair_num = len(repair_operators)
        for r in repair_operators:
            alns.add_repair_operator(r)

        # initial temperatures can be autofitted such that the frist solution has a 50% chance of being acceted?
        max_iterations = 5000000
        op_coupling = np.zeros((destroy_num, repair_num))
        for i in range(destroy_num):
            for j in range(repair_num):
                if i == j:
                    op_coupling[i, j] = True
                elif j == (repair_num - 1) and i == (destroy_num - 1):
                    op_coupling[i, j] = True
                else:
                    op_coupling[i, j] = False

        # select = RouletteWheel([25, 15, 5, 0], 0.6, destroy_num, repair_num, op_coupling)
        select = MABSelector(
            [25, 15, 5, 0],
            destroy_num,
            repair_num,
            learning_policy=LearningPolicy.EpsilonGreedy(0.2),
            op_coupling=op_coupling,
        )
        accept = SimulatedAnnealing.autofit(
            curr_state.objective(),
            epsilon,
            accept_prob,
            max_iterations,
            method="exponential",
        )
        # stop = MaxRuntime(98)
        stop = NoImprovement(1000)
        result = alns.iterate(curr_state, select, accept, stop)

        counts = result.statistics.destroy_operator_counts
        # Iterate over the repair operator counts
        for operator, outcome_counts in counts.items():
            print(f"Operator: {operator}")
            for i, count in enumerate(outcome_counts):
                print(f"Outcome {i+1}: {count}")

        curr_state = result.best_state
    return curr_state
