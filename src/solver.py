from vrp_parser import VRPInstance
import numpy as np
import math
import copy as cp

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import NoImprovement
from alns.select import RouletteWheel
from alns.select import OperatorSelectionScheme

from itertools import permutations

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
        self.unassigned_customers = set()  #this is used for global destroy and repair operators
        self.removed_customers = {} #used for local operatores, where you still want to keep them in the same route

    def copy(self):
        return VRPState(cp.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        #computing the distance calculated by all vehicles
        cost = 0
        distance = 0
        #completely ignore anything that goes past max_num_vehicles
        if (self.num_vehicles > self.max_num_vehicles):
            return float('inf')
        
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
                    distance += math.sqrt(self.customer_x[curr_customer]**2 + self.customer_y[curr_customer ]**2)
                else:
                    #now that we are at this client, how costly was it to get here
                    prev_customer = customer_list[j-1]
                    distance += math.sqrt((self.customer_x[curr_customer] - self.customer_x[prev_customer ])**2 + (self.customer_y[curr_customer ] - self.customer_y[prev_customer])**2)
            
            #getting back to the original point (0,0)
            if len(customer_list )> 0:
                last_customer = customer_list[-1]
                distance += math.sqrt(self.customer_x[last_customer]**2 + self.customer_y[last_customer ]**2) #getting back to the lot
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


def two_opt_repair(state: VRPState, rnd_state, **kwargs): #this is like best local repair
    cp_state = cp.deepcopy(state)
    to_insert = cp.deepcopy(cp_state.removed_customers)
    cp_state.removed_customers.clear()
    for car, customers in to_insert.items():
        start = customers[0]
        end = customers[1]
        cp_state.vehicle_to_route[car][start:end] = cp_state.vehicle_to_route[car][start:end][::-1] 
    return cp_state

    

def find_cost(state, route):
    distance = 0
    for j in range(len(route)):
        curr_customer = route[j]
        if j == 0 :
            distance += math.sqrt(state.customer_x[curr_customer]**2 + state.customer_y[curr_customer ]**2)
        else:
            #now that we are at this client, how costly was it to get here
            prev_customer = route[j-1]
            distance += math.sqrt((state.customer_x[curr_customer] - state.customer_x[prev_customer ])**2 + (state.customer_y[curr_customer ] - state.customer_y[prev_customer])**2)
        #getting back to the original point (0,0)
        if len(route)> 0:
            last_customer = route[-1]
            distance += math.sqrt(state.customer_x[last_customer]**2 + state.customer_y[last_customer ]**2) #getting back to the lot
        return distance
            

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
        best_cost, best_car, best_pos = float('inf'), None, None
        for car, route in state.vehicle_to_route.items(): #if this route can service the customer
            if cp_state.vehicle_to_capacity[car] >= cp_state.customer_demand[unassigned]:
                # cost to insert a customer at a given point in the program
                if (len(route)) == 0:
                    cost = math.sqrt((x_coord )**2 + (y_coord)**2) * 2 #times 2 because the car has to go back and forth
                    if cost < best_cost:
                        best_cost = cost
                        best_car = car
                        best_pos = 0
                else:
                    for i in range(len(route)):
                        cost = None
                        if i== 0 or i == len(route)-1:
                            #cost from this current node to either the current first or current last
                            cost = math.sqrt((x_coord - cp_state.customer_x[route[i]])**2 + (y_coord - cp_state.customer_y[route[i]])**2)
                            #now getting to the new first or getting back to the initial from first                           
                            cost +=  math.sqrt((x_coord )**2 + (y_coord)**2) 
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

#destroy operators
def two_opt_remove(state : VRPState, rnd_state, ):
    #remove two random edges and try best insertion back
    cp_state = cp.deepcopy(state)
    cp_state.removed_customers.clear()
    for car,route in state.vehicle_to_route.items():
        if (len(route) <= 2):
            continue
        length = rnd_state.choice(np.arange(0, len(route)-1, 1), 1, replace=False)[0]
        start = rnd_state.choice(np.arange(0, len(route) - length), 1, replace=False)[0]
        end = start + length
        cp_state.removed_customers[car] = [start,end]
    return cp_state

def random_removal(state : VRPState, rnd_state, n_remove=1):
    cp_state = cp.deepcopy(state)
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
            cp_state.unassigned_customers.add(removed_customer)
            route.remove(removed_customer)
            #update capacity for the vehicles whose customers have been removed
            cp_state.vehicle_to_capacity[car] += cp_state.customer_demand[removed_customer]
    return cp_state

def begin_search(vrp_instance):
    seed = np.random.randint(1,1000000)
    alns = ALNS(np.random.RandomState(seed))

    initial_veh_to_customer, initial_num_vehicles, vehicle_to_capacity, unassigned = vrp_instance.construct_intial_solution()
    initial_state = VRPState(vrp_instance, initial_veh_to_customer, initial_num_vehicles,vehicle_to_capacity )
    #the initial solution might not have used up all cars
    initial_state.num_vehicles = len(initial_state.vehicle_to_route)
    initial_state.fake_vehicle_customers = unassigned
    
    #add destroy and repair operators
    destroy_num = 2
    repair_num = 2
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(two_opt_remove)
    alns.add_repair_operator(best_global_repair)
    alns.add_repair_operator(two_opt_repair)
    

    #initial temperatures can be autofitted such that the frist solution has a 50% chance of being acceted?
    max_iterations = 5000
    op_coupling = np.zeros((destroy_num, repair_num))
    op_coupling[0][0] = True
    op_coupling[0][1] = False 
    op_coupling[1][0] = False
    op_coupling[1][1] = True
    select = RouletteWheel([25, 15, 5, 0], 0.5, destroy_num, repair_num, op_coupling)
    # select = OperatorSelectionScheme(destroy_num, repair_num,op_coupling)
    accept = SimulatedAnnealing.autofit(initial_state.objective(), 0.05, 0.80, max_iterations)
    stop = NoImprovement(max_iterations= 100) #this 20 was a random choice
    result = alns.iterate(initial_state, select, accept, stop)

    return result.best_state

    
            