from alns import ALNS
from vrp_parser import VRPInstance
import numpy as np
import math
import copy as cp

class Solver:
    def __init__ (self, vrp_instance: VRPInstance):
        self.vrp_instance = vrp_instance

    class VRPState:
        """
        Solution state for CVRP. It has two data members, routes and unassigned.
        Routes is a list of list of integers, where each inner list corresponds to
        a single route denoting the sequence of customers to be visited. A route
        does not contain the start and end depot. Unassigned is a list of integers,
        each integer representing an unassigned customer.
        """

        def __init__(self,vrp_instance : VRPInstance, vehicles_to_route, num_vehicles):
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
            for i in range(self.num_customers):
                unserved_customers.add(i)
            for i in range(self.num_vehicles):
                customer_list = self.vehicle_to_route[i]
                capacity_served = 0
                for j in range(len(customer_list)):
                    unserved_customers.remove(j)
                    capacity_served += self.customer_demand[j]
                    if j == 0 == (len(customer_list) -2):
                        distance += math.sqrt(self.customer_x[j]**2 + self.customer_y[j]**2)
                    else:
                        distance += math.sqrt((self.customer_x[j] - self.customer_x[j+1])**2 + (self.customer_y[j] - self.customer_y[j+1])**2)
                #if we exceed capaity at any point also ignore it
                if capacity_served > self.vehicle_capacity:
                    return float('inf')
            cost += distance
            #if we have some customers that we are not serving
            if len(unserved_customers) > 0:
                return float('inf')
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
            for idx, route in self.vehicle_to_customers:
                if customer in route:
                    return route

            raise ValueError(f"Solution does not contain customer {customer}.")

        #repair operators
        def two_opt_repair(self, state, vehicle_idx,customer_idx):
            cp_state = cp.deepcopy(state)



        #destroy operators
        def two_opt_remove(self, state):
            
        

    def main(self):
        seed = np.random.int(1,1000000)
        alns = ALNS(np.rnd.RandomState(seed))

        initial_veh_to_customer, initial_num_vehicles = self.vrp_instance.construct_intial_solution()
        initial_state = self.VRPState(self.vrp_instance, initial_veh_to_customer, initial_num_vehicles)
        #the initial solution might not have used up all cars
        initial_state.num_vehicles = len(initial_state.vehicle_to_route)

        #add destroy and repair operators

        select = None
        start_temperature = 0
        end_temperature = 0
        accept = alns.accept.SimulatedAnnealing.SimulatedAnnealing(start_temperature, end_temperature, step =0.5, method ='exponential')
        stop = alns.stop.NoImprovement.NoImprovement(max_iterations= 20) #this 20 was a random choice
        result = alns.iterate(initial_state, select, accept, stop)

        return result

    if __name__ == "__main__":
    # Call the main function
        main()
        
            