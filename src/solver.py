from vrp_parser import VRPInstance
import numpy as np
import math
import copy as cp
from collections import defaultdict


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

    def __init__(self,vrp_instance : VRPInstance, vehicles_to_route, num_vehicles, vehicle_to_capacity ):
        self.vrp_instance = vrp_instance
        self.max_num_vehicles = self.vrp_instance.num_vehicles
        self.num_customers = self.vrp_instance.num_customers
        self.vehicle_capacity = self.vrp_instance.vehicle_capacity
        self.customer_demand = self.vrp_instance.demandOfCustomer
        self.customer_x = self.vrp_instance.xCoordOfCustomer
        self.customer_y = self.vrp_instance.yCoordOfCustomer

        #what makes a solution different
        self.num_vehicles = num_vehicles
        self.vehicle_to_route = vehicles_to_route
        self.vehicle_to_capacity = vehicle_to_capacity 
        self.fake_vehicle_customers = set() #hold customers who cannot be added back to the routes after they were removed
        self.unassigned_customers = [] #this is used for global destroy and repair operators
        self.removed_customers = {} #used for local operatores, where you still want to keep them in the same route

    def copy(self):
        pass
    
    def objective(self):
        #computing the distance calculated by all vehicles
        cost = 0
        distance = 0
        #completely ignore anything that goes past max_num_vehicles
        if (self.num_vehicles > self.max_num_vehicles):
            return float('inf')
    
        depot_x = self.customer_x[0]
        depot_y = self.customer_y[0]
        
        unserved_customers = set()
        for i in range(1,self.num_customers):
            unserved_customers.add(i)
        for i in range(self.num_vehicles):
            customer_list = self.vehicle_to_route[i]
            capacity_served = 0
            #for each route calculate the distance traveled
            for j in range(len(customer_list)):
                curr_customer = customer_list[j]
                unserved_customers.remove(curr_customer )
                capacity_served += self.customer_demand[curr_customer]
                if j == 0 :
                    distance += math.sqrt((self.customer_x[curr_customer] - depot_x)**2 + (self.customer_y[curr_customer]-depot_y)**2)
                else:
                    #now that we are at this client, how costly was it to get here
                    prev_customer = customer_list[j-1]
                    distance += math.sqrt((self.customer_x[curr_customer] - self.customer_x[prev_customer ])**2 + (self.customer_y[curr_customer ] - self.customer_y[prev_customer])**2)
            
            #getting back to the original point (0,0)
            if len(customer_list )> 0:
                last_customer = customer_list[-1]
                distance += math.sqrt((self.customer_x[last_customer] - depot_x)**2 + (self.customer_y[last_customer] - depot_y)**2) #getting back to the lot
            #if we exceed capaity at any point also ignore it
            if capacity_served > self.vehicle_capacity:
                return float('inf')
        cost += distance

        # we don't inmediately return infinite here since we believe that routes with fewer customers in the fake vehicle are closer to the true solution
        cost += len(self.fake_vehicle_customers) * 100000
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
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")


#this is basically changing two edges by taking some middle chunk and flipping it so that the conenctin parts are diff
def switch_repair(state: VRPState, rnd_state, **kwargs): #this is like best local repair
    cp_state = cp.deepcopy(state)
    to_insert = cp.deepcopy(cp_state.removed_customers)
    cp_state.removed_customers.clear()
    cp_state.unassigned_customers.clear()
    for car, customers in to_insert.items():
        start = customers[0]
        end = customers[1]
        cp_state.vehicle_to_route[car][start:end] = cp_state.vehicle_to_route[car][start:end][::-1] 

    return cp_state

def switch_remove(state : VRPState, rnd_state, **kwargs):#this switch is within one route
    #remove two random edges and try best insertion back
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    for car,route in state.vehicle_to_route.items():
        if (len(route) <= 2):
            continue
        length = rnd_state.choice(np.arange(0, len(route)), 1, replace=False)[0]
        
        start = rnd_state.choice(np.arange(0, len(route) - length), 1, replace=False)[0]
        end = start + length
        cp_state.removed_customers[car] = [start,end]
    return cp_state
 
#helper for the above function 
def find_cost(state : VRPState, route):
    distance = 0
    depot_x = state.customer_x[0]
    depot_y = state.customer_y[0]
    for j in range(len(route)):
        curr_customer = route[j]
        if j == 0 :
            distance += math.sqrt((state.customer_x[curr_customer] - depot_x)**2 + (state.customer_y[curr_customer]-depot_y)**2)
        else:
            #now that we are at this client, how costly was it to get here
            prev_customer = route[j-1]
            distance += math.sqrt((state.customer_x[curr_customer] - state.customer_x[prev_customer ])**2 + (state.customer_y[curr_customer ] - state.customer_y[prev_customer])**2)
        #getting back to the original point (0,0)
        if len(route)> 0:
            last_customer = route[-1]
            distance += math.sqrt(state.customer_x[last_customer]**2 + state.customer_y[last_customer ]**2) #getting back to the lot
    return distance

#this is for blocks of customers (a segment)
def switch_across_routes(state : VRPState, rnd_state, **kwargs): #switches segments across routes
    cp_state = cp.deepcopy(state)
    num_destroy = max(2, len(state.vehicle_to_route)//2)
    if num_destroy % 2 != 0:
        num_destroy += 1
    to_destroy = list(rnd_state.choice(len(state.vehicle_to_route)-1, num_destroy, replace = False))
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    valid = True
    for car in to_destroy:
        if len(state.vehicle_to_route[car]) < 2:
            valid = False
    if not valid:
        return cp_state
    
    while (len(to_destroy) > 0):
        r = rnd_state.choice(to_destroy, 1, replace = False)[0]  
        route = cp_state.vehicle_to_route[r]
        r_length = len(cp_state.vehicle_to_route[r])
        # if r_length < 2:
        #     to_destroy.remove(r)
        #     continue
        start = rnd_state.choice(np.arange(1,r_length), 1, replace = False)[0]
        to_remove = route[:start]
        cp_state.removed_customers[r] = to_remove
        to_destroy.remove(r)
        cp_state.vehicle_to_route[r] = route[start:]
    return cp_state

def insert_across_routes(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    segments = cp.deepcopy(state.removed_customers)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    while len(segments) > 0:
        segment_pair_idx = rnd_state.choice(list(segments.keys()), 2, replace = False)
        veh1 = segment_pair_idx[0]
        veh2 = segment_pair_idx[1]
        segment_pair = [segments[veh1], segments[veh2]]
        #swtich between routes by changing the beginnings (this always get adversely not selected so there might be something wrong idk)
        cp_state.vehicle_to_route[veh1] = segment_pair[1] + cp_state.vehicle_to_route[veh1]
        cp_state.vehicle_to_route[veh2] = segment_pair[0] + cp_state.vehicle_to_route[veh2]
        for s in segment_pair_idx:
            segments.pop(s)
    return cp_state

def remove_furthest_out(state : VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()

    def find_farthest_customer(state : VRPState,customers):
        farthest_customer = None
        farthest_dist = float('-inf')
        #skipping the first one, initial cost of getting out there
        x = state.customer_x[customers[0]]
        y = state.customer_y[customers[0]]
        for i in range(1, len(customers)):
            c = customers[i]
            dist = math.sqrt((state.customer_x[c] - x) ** 2 + (state.customer_y[c]- y)**2)
            if dist > farthest_dist:
                farthest_dist = dist
                farthest_customer = c
            x = state.customer_x[c]
            y = state.customer_y[c]
        #if the start and end are too far apart (going and getting back gets too expensive)
        if math.sqrt((state.customer_x[customers[0]] - state.customer_x[len(customers)-1]) ** 2 + (state.customer_y[customers[0]]- state.customer_x[len(customers)-1])**2) > farthest_dist:
            return customers[len(customers)-1]
        return farthest_customer
    for c,r in state.vehicle_to_route.items():
        if len(r) < 2:
            continue
        farthest = find_farthest_customer(state, r)
        cp_state.unassigned_customers.append(farthest)
        cp_state.vehicle_to_route[c].remove(farthest)
    return cp_state


def random_removal(state : VRPState, rnd_state, n_remove=1):
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.removed_customers.clear()
    for car, route in cp_state.vehicle_to_route.items():
        if len(route) == 0:
            continue
        if len(route) < 2:
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
            #update capacity for the vehicles whose customers have been removed
            cp_state.vehicle_to_capacity[car] += cp_state.customer_demand[removed_customer]
    return cp_state

def best_global_repair(state: VRPState, rnd_state, **kwargs):
    #check if there is a better place to insert the customer in
    cp_state = cp.deepcopy(state)
    cp_state.unassigned_customers.clear()
    cp_state.fake_vehicle_customers.clear()
    total_unassigned = set()
    for c in state.unassigned_customers:
        total_unassigned.add(c)
    for c in state.fake_vehicle_customers:
        total_unassigned.add(c)

    for unassigned in total_unassigned:
        x_coord = cp_state.customer_x[unassigned]
        y_coord = cp_state.customer_y[unassigned]
        depot_x = state.customer_x[0]
        depot_y = state.customer_y[0]
        best_cost, best_car, best_pos = float('inf'), None, None
        for car, route in state.vehicle_to_route.items(): #if this route can service the customer
            if cp_state.vehicle_to_capacity[car] >= cp_state.customer_demand[unassigned]:
                # cost to insert a customer at a given point in the program
                if (len(route)) == 0:
                    cost = math.sqrt((x_coord - depot_x)**2 + (y_coord - depot_y)**2) * 2 #times 2 because the car has to go back and forth
                    if cost < best_cost:
                        best_cost = cost
                        best_car = car
                        best_pos = 0
                else:
                    for i in range(len(route)):
                        cost = 0
                        if i == 0 or i == len(route)-1:
                            #cost from this current node to either the current first or current last
                            cost = math.sqrt((x_coord - cp_state.customer_x[route[i]])**2 + (y_coord - cp_state.customer_y[route[i]])**2)
                            #now getting to the new first or getting back to the initial from first                           
                            cost +=  math.sqrt((x_coord - depot_x)**2 + (y_coord - depot_y)**2) 
                        else:
                            #from the current node to this new one
                            cost = math.sqrt((x_coord - cp_state.customer_x[route[i]])**2 + (y_coord - cp_state.customer_y[route[i]])**2)
                            #from the new one back to the +1 that it was connected to
                            cost += math.sqrt((x_coord - cp_state.customer_x[route[i+1]])**2 + (y_coord - cp_state.customer_y[route[i+1]])**2)
                        if cost < best_cost:
                            best_cost = cost
                            best_car = car
                            best_pos = i+1
        #it might be the case that the nodes we removed cannot be properly reinserted into the routes given the order in which we go through them, in that case we don't want to consider this a good solution
        if best_pos == None:
            cp_state.fake_vehicle_customers.add(unassigned)
        else:
            cp_state.vehicle_to_route[best_car].insert(best_pos, unassigned) #python insert displaces the item in that current position and pushes everything back
            cp_state.vehicle_to_capacity[best_car] -= cp_state.customer_demand[unassigned]
    return cp_state

def reorder_everything(state : VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)
    for car, route in state.vehicle_to_route.items():
        cp_state.vehicle_to_route[car] = set(route)
    return cp_state

def greedy_each(state: VRPState, rnd_state, **kwargs):
    cp_state = cp.deepcopy(state)

    def find_closest_customer(car_x, car_y, customers):
        closest_customer = None
        closest_dist = float('inf')
        for c in customers:
            dist = math.sqrt((state.customer_x[c] - car_x) ** 2 + (state.customer_y[c]- car_y)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_customer = c
        return closest_customer
        
    for car, route in state.vehicle_to_route.items():
        unserved_customers = cp.deepcopy(route)
        new_ordering = []
        cx = state.customer_x[0]
        cy = state.customer_y[0]
        while len(unserved_customers) > 0:
            new_customer = find_closest_customer(cx, cy, unserved_customers)
            unserved_customers.remove(new_customer)
            new_ordering.append(new_customer)
            cx = state.customer_x[new_customer]
            cy = state.customer_y[new_customer]
        cp_state.vehicle_to_route[car] = new_ordering
    return cp_state



def begin_search(vrp_instance):
    seed = np.random.randint(1,1000000)

    #acceptance thingy, as we go down we only allow a lower "worse" than current solution
    epsilons = [0.2, 0.1, 0.05] 
    accept_probs = [0.95, 0.75, 0.5] #probability we accept a solution at most epsilon percentage worse than current best
    initial_veh_to_customer, initial_num_vehicles, vehicle_to_capacity, unassigned = vrp_instance.construct_intial_solution()
    initial_state = VRPState(vrp_instance, initial_veh_to_customer, initial_num_vehicles,vehicle_to_capacity )
    #the initial solution might not have used up all cars
    initial_state.num_vehicles = len(initial_state.vehicle_to_route)
    initial_state.fake_vehicle_customers = unassigned
    curr_state = initial_state
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        accept_prob = accept_probs[i]
        alns = ALNS(np.random.RandomState(seed))
        #add destroy and repair operators
        destroy_num = 4
        repair_num = 4
        alns.add_destroy_operator(random_removal)
        alns.add_destroy_operator(switch_remove)
        alns.add_destroy_operator(switch_across_routes)
        alns.add_destroy_operator(reorder_everything)
        # alns.add_destroy_operator(switch_across_routes_one)
        # alns.add_destroy_operator(remove_furthest_out)

        alns.add_repair_operator(best_global_repair)
        alns.add_repair_operator(switch_repair)
        alns.add_repair_operator(insert_across_routes)
        alns.add_repair_operator(greedy_each)
        # alns.add_repair_operator(insert_across_routes_one)
        #initial temperatures can be autofitted such that the frist solution has a 50% chance of being acceted?
        max_iterations = 100000
        op_coupling = np.zeros((destroy_num, repair_num))
        for i in range(destroy_num):
            for j in range(repair_num):
                if i == j:
                    op_coupling[i,j] = True
                # elif i == (destroy_num - 1 )and j == 0:
                #     op_coupling[i,j] = True
                else:
                    op_coupling[i,j] = False
        # select = RouletteWheel([25, 15, 5, 0], 0.8, destroy_num, repair_num, op_coupling)
        select =  MABSelector([25,15,5,0], destroy_num, repair_num, learning_policy = LearningPolicy.EpsilonGreedy(epsilon=0.2),op_coupling = op_coupling)
        accept = SimulatedAnnealing.autofit(curr_state.objective(), epsilon, accept_prob, max_iterations, method = 'exponential')
        # stop = MaxRuntime(100) 
        stop = NoImprovement(1000)
        result = alns.iterate(initial_state, select, accept, stop)

        counts = result.statistics.repair_operator_counts
        # Iterate over the repair operator counts
        for operator, outcome_counts in counts.items():
            print(f"Operator: {operator}")
            for i, count in enumerate(outcome_counts):
                print(f"Outcome {i+1}: {count}")

        curr_state = result.best_state
    return curr_state

    
            