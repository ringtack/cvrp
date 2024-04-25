import alns
from vrp_parser import VRPInstance
import numpy as np
import math

class Solver:
    def __init__ (self,vrp_instance : VRPInstance):
        self.vrp_instance = vrp_instance
        self.num_vechiles = self.vrp_instance.num_vehicles
        self.num_customers = self.vrp_instance.num_customers
        self.vehicle_capacity = self.vrp_instance.vehicle_capacity
        self.customer_demand = self.vrp_instance.demandOfCustomer
        self.customer_x = self.vrp_instance.xCoordOfCustomer
        self.customer_y = self.vrp_instance.yCoordOfCustomer
        self.vehicle_to_customers = {}
        self.vehicle_to_capacity = {}
        for i in range(self.num_vechiles):
            self.vehicle_to_customers[i] = []
            self.vehicle_to_capacity[i] = self.vehicle_capacity
        

    def construct_intial_solution(self):
        unserved_customers = set()
        for i in range(self.customers):
            unserved_customers.add(i)
        
        vehicle_num = 0
        while (len(unserved_customers) > 0):
            customer_idx = -1
            while (customer_idx == -1):
                x_pos = -1
                y_pos = -1
                #no customers visited so far
                if len(self.vehicle_to_customers[vehicle_num][-1]):
                    x_pos = 0
                    y_pos = 0
                else:
                    last_customer_idx = self.vehicle_to_customers[vehicle_num][-1]
                    x_pos = self.customer_x[last_customer_idx]
                    y_pos = self.customer_y[last_customer_idx]
                customer_idx = self.find_closest_customer(self, x_pos, y_pos, unserved_customers, self.vehicle_to_capacity[vehicle_num])
                #if this vehicle cannot serve *any* customer it means that it has no spare capacity
                if (customer_idx == -1):
                    vehicle_num += 1
            unserved_customers.remove(customer_idx)
            self.vehicle_to_capacity[vehicle_num] -= self.customer_demand[customer_idx]
            self.vehicle_to_customers[vehicle_num].apend(customer_idx)

    def find_closest_customer(self, vehicle_x, vehicle_y, unserved_customers, capacity):
        closest_idx = -1
        closest_distance = float.max('inf')
        for customer in unserved_customers:
            distance = math.sqrt((vehicle_x - self.customer_x[customer])**2 + (vehicle_y - self.customer_x[customer])**2)
            if self.customer_demand[customer] <= capacity and distance < closest_distance:
                closest_distance = distance
                closest_distance = customer
        return closest_idx

    
    class CustomerInfo:

        def __init__ (self,demand, x, y):
            self.demand = demand
            self.x = x
            self.y = y
    
        