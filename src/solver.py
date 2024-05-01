from vrp_parser import VRPInstance
import numpy as np
import math
import copy as cp

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import NoImprovement
from alns.select import RouletteWheel

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
        self.unassigned_customers = set()

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
            if len(customer_list )> 0:
                last_customer = customer_list[-1]
                distance += math.sqrt(self.customer_x[last_customer]**2 + self.customer_y[last_customer ]**2) #getting back to the lot
            #if we exceed capaity at any point also ignore it
            if capacity_served > self.vehicle_capacity:
                return float('inf')
        cost += distance
        #if we have some customers that we are not serving
        if len(unserved_customers) > 0:
            return float('inf')
        print(cost)
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


def two_opt_repair(state): #this is like best local repair
    cp_state = cp.deepcoppy(state)
    for unassigned in state.unassigned_customers:
        x_coord = cp_state.customer_x[unassigned]
        y_coord = cp_state.customer_y[unassigned]
        best_cost, best_pos = float('inf'), -1
        route = state.find_route(unassigned)
        for i in range(len(route)):
            if i==0 or i == len(route)-1:
                cost = math.sqrt((x_coord - cp_state.customer_x[route[i]])**2 + (y_coord - cp_state.customer_y[route[i]])**2)
            else:
                #from the current node to this new one
                cost = math.sqrt((x_coord - cp_state.customer_x[route[i]])**2 + (y_coord - cp_state.customer_y[route[i]])**2)
                #from the new one back to the +1 that it was connected to
                cost = math.sqrt((x_coord - cp_state.customer_x[route[i+1]])**2 + (y_coord - cp_state.customer_y[route[i+1]])**2)
            if cost < best_cost:
                best_cost = cost
                best_route = route
                best_pos = i+1
        cp_state.unassigned.remove(unassigned)
        cp_state.vehicle_to_route[best_route].insert(best_pos) #python insert displaces the item in that current position and pushes everything back
        cp_state.vehicle_to_capacity[best_route] -= state.customer_demand[unassigned]

def best_global_repair(state: VRPState, rnd_state, **kwargs):
    #check if there is a better place to insert the customer in
    cp_state = cp.deepcopy(state)
    for unassigned in state.unassigned_customers:
        x_coord = cp_state.customer_x[unassigned]
        y_coord = cp_state.customer_y[unassigned]
        best_cost, best_car, best_pos = float('inf'), -1, -1
        for car, route in state.vehicle_to_route.items(): #if this route can service the customer
            if state.vehicle_to_capacity[car] >= state.customer_demand[unassigned]:
                # cost to insert a customer at a given point in the program
                if (len(route)) == 0:
                    cost = math.sqrt((x_coord )**2 + (y_coord)**2) * 2 #times 2 because the car has to go back and forth
                    if cost < best_cost:
                        best_cost = cost
                        best_car = car
                        best_pos = 0
                else:
                    for i in range(len(route)):
                        if i==0 or i == len(route)-1:
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
        cp_state.unassigned_customers.remove(unassigned)
        cp_state.vehicle_to_route[best_car].insert(best_pos, unassigned) #python insert displaces the item in that current position and pushes everything back
        cp_state.vehicle_to_capacity[best_car] -= state.customer_demand[unassigned]
    return cp_state

#destroy operators
def two_opt_remove(state):
    cp_state = cp.deepcopy(state)
    for route in cp_state:
        to_remove_idx = np.random.randint(0, len(route))
        removed_customer = route[to_remove_idx]
        cp_state.unassigned_customers.add(removed_customer)
        #don't actually remove from route because we need to figure out which one it is in
    return cp_state

def random_removal(state : VRPState, rnd_state, n_remove=1):
    cp_state = cp.deepcopy(state)
    print(state.vehicle_to_route)
    for car, route in cp_state.vehicle_to_route.items():
        if len(route) == 0:
            continue
        if len(route) < 2:
            n_remove = 1
        else:
            n_remove = 2
        rnd_state.choice(len(route), n_remove, replace=False)
        to_remove_idx = np.random.randint(0, len(route))
        removed_customer = route[to_remove_idx]
        cp_state.unassigned_customers.add(removed_customer )
        route.remove(removed_customer)
        cp_state.vehicle_to_capacity[car] += cp_state.customer_demand[removed_customer]
    return cp_state

def begin_search(vrp_instance):
    seed = np.random.randint(1,1000000)
    alns = ALNS(np.random.RandomState(seed))

    initial_veh_to_customer, initial_num_vehicles, vehicle_to_capacity = vrp_instance.construct_intial_solution()
    initial_state = VRPState(vrp_instance, initial_veh_to_customer, initial_num_vehicles,vehicle_to_capacity )
    #the initial solution might not have used up all cars
    initial_state.num_vehicles = len(initial_state.vehicle_to_route)
    
    #add destroy and repair operators
    alns.add_destroy_operator(random_removal)
    # alns.add_destroy_operator(self.two_opt_remove)
    alns.add_repair_operator(best_global_repair)
    # alns.add_repair_operator(self.two_opt_repair)
    

    #initial temperatures can be autofitted such that the frist solution has a 50% chance of being acceted?
    start_temperature = 0
    end_temperature = 1
    max_iterations = 5000
    # op_coupling = [True, False ; False, True] 
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    accept = SimulatedAnnealing.autofit(initial_state.objective(), 0.05, 0.90, max_iterations)
    print("initially")
    print(initial_state.objective())
    stop = NoImprovement(max_iterations= 20) #this 20 was a random choice
    print("started")
    result = alns.iterate(initial_state, select, accept, stop)

    return result

    
            