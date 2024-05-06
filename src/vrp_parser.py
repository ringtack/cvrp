import math
import copy as cp

class VRPInstance:
    def __init__(self, fileName):
        try:
            with open(fileName, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError as e:
            print("Error: in VRPInstance() " + fileName + "\n" + str(e))
            exit(-1)

        # Parsing input parameters
        input_params = lines[0].split()
        self.num_customers = int(input_params[0])
        self.num_vehicles = int(input_params[1])
        self.vehicle_capacity = int(input_params[2])

        print("Number of customers:", self.num_customers)
        print("Number of vehicles:", self.num_vehicles)
        print("Vehicle capacity:", self.vehicle_capacity)

        self.demandOfCustomer = []
        self.xCoordOfCustomer = []
        self.yCoordOfCustomer = []
        for line in lines[1:]: #last line is an empty string
            data = line.split()
            if len(data) == 0:
                continue
            self.demandOfCustomer.append(int(data[0]))
            self.xCoordOfCustomer.append(float(data[1]))
            self.yCoordOfCustomer.append(float(data[2]))
        for i in range(self.num_customers):
            print(self.demandOfCustomer[i], self.xCoordOfCustomer[i], self.yCoordOfCustomer[i])
        

    def get_distance_between(self, i,j):
        return math.sqrt((self.xCoordOfCustomer[i] - self.xCoordOfCustomer[j])**2 + (self.yCoordOfCustomer[i] - self.yCoordOfCustomer[j])**2) #getting back to the lot

    def get_smallest_customer(self,unserved, position, capacity):
        smallest = None
        smallest_d = float('inf')
        for c in unserved:
            if capacity >= self.demandOfCustomer[c] and self.demandOfCustomer[c] < smallest_d:
                    smallest = c
                    smallest_d = self.demandOfCustomer[c]
        return smallest
    
    def construct_intial_solution(self):
        unserved_customers = set()
        for i in range(1,self.num_customers):
            unserved_customers.add(i)
        
        vehicle_to_customers = {}
        vehicle_to_capacity = {}

        
        def find_route(customer):
            for v, route in vehicle_to_customers.items():
                if customer in route:
                    return v
            return None
        
        def find_demand(route):
            d= 0
            for c in route:
                d += self.demandOfCustomer[c]
            return d

        #first let's serve all customers separately
        costs = []
        for c in range(1,self.num_customers):
            costs.append(2 * self.get_distance_between( c,0))
        savings = {}
        for i in range(1, self.num_customers):
            for j in range(i, self.num_customers):
                if i == j:
                    continue
                savings[(i,j)] = self.get_distance_between(i,0) + self.get_distance_between(j,0) - self.get_distance_between(i,j)
        sorted_savings = dict(sorted(savings.items(), key=lambda x: (x[1], self.demandOfCustomer[x[0][0]] + self.demandOfCustomer[x[0][1]] ), reverse=True))
        for pair, v in sorted_savings.items():
            i = pair[0]
            j = pair[1]
            if i in unserved_customers and j in unserved_customers:
                if self.demandOfCustomer[i] + self.demandOfCustomer[j] > self.vehicle_capacity:
                    continue
                if len(vehicle_to_customers) >= self.num_vehicles: #cannot add more
                    continue
                new_idx = None
                if len(vehicle_to_customers) == 0:
                    new_idx = 0
                else:
                    new_idx, _ = list(vehicle_to_customers.items())[-1] 
                    new_idx += 1
                vehicle_to_customers[new_idx] = []
                vehicle_to_capacity[new_idx] = self.vehicle_capacity
                idx, _ = list(vehicle_to_customers.items())[-1]
                route = vehicle_to_customers[idx]
                route.append(i)
                route.append(j)
                vehicle_to_capacity[idx] -= (self.demandOfCustomer[i] + self.demandOfCustomer[j])
                unserved_customers.remove(i)
                unserved_customers.remove(j)
            #both are in some other route
            elif (i not in unserved_customers) and (j not in unserved_customers):
                i_route_num = find_route(i)
                i_route = vehicle_to_customers[i_route_num]
                j_route_num = find_route(j)
                j_route = vehicle_to_customers[j_route_num]
                if i_route_num == j_route_num:
                    continue
                idx_i = i_route.index(i)
                idx_j = j_route.index(j)
                demand_j = find_demand(vehicle_to_customers[j_route_num])
                demand_i = find_demand(vehicle_to_customers[i_route_num])
                if (idx_i == 0 or idx_i == len(j_route)-1) and (idx_j == 0 or idx_j == len(j_route)-1):
                    if demand_j + demand_i <= self.vehicle_capacity:
                        new_route = None
                        if idx_i == 0 and idx_j == 0:
                            to_add = j_route[::-1]
                            new_route = to_add + i_route
                        elif idx_i == 0 and idx_j == len(j_route)-1:
                            new_route= j_route + i_route
                        elif idx_i == len(i_route) -1 and idx_j == 0:
                            new_route = i_route + j_route
                        else:
                            new_route = j_route[::-1] + i_route[::-1]
                        vehicle_to_customers.pop(j_route_num)
                        vehicle_to_capacity.pop(j_route_num)
                        vehicle_to_customers[i_route_num] = new_route
                        vehicle_to_capacity[i_route_num] = self.vehicle_capacity - (demand_j + demand_i)
                
            else:
                i_route_num = find_route(i)
                j_route_num = find_route(j)
                unassigned = None
                assigned = None
                idx_assigned = None
                assigned_v = None
                if i_route_num == None:
                    unassigned = i
                    assigned = j
                    idx_assigned = vehicle_to_customers[j_route_num].index(j)
                    assigned_v = j_route_num
                else:
                    unassigned = j
                    assigned = i
                    idx_assigned = vehicle_to_customers[i_route_num].index(i)
                    assigned_v = i_route_num
                if idx_assigned == 0 or idx_assigned == (len(vehicle_to_customers[assigned_v])-1):
                    if self.demandOfCustomer[unassigned] <= vehicle_to_capacity[assigned_v]:
                        if idx_assigned == 0:
                            vehicle_to_customers[assigned_v].insert(idx_assigned,unassigned)
                        else:
                            vehicle_to_customers[assigned_v].append(unassigned)
                        unserved_customers.remove(unassigned)
                        vehicle_to_capacity[assigned_v] -= self.demandOfCustomer[unassigned]
        
        routes_copy = cp.deepcopy(vehicle_to_customers)
        cap_copy = cp.deepcopy(vehicle_to_capacity)
        vehicle_to_customers.clear()
        vehicle_to_capacity.clear()
        idx = 0
        for v, r in routes_copy.items():
            vehicle_to_customers[idx] = r
            vehicle_to_capacity[idx] = cap_copy[v]
            idx += 1
        
        #what cannot be down smartly will be done greedily
        print(unserved_customers)
        demands = {}
        for c in unserved_customers:
            demands[c] = self.demandOfCustomer[c]
        ordered_unserved = dict(sorted(demands.items(), key=lambda x: x[1], reverse = True))
        ordered_unserved = list(ordered_unserved.keys())
        #as long as there are unassigned and we have more cars that have not been used yet
        while (len(ordered_unserved) > 0 and len(vehicle_to_customers) < self.num_vehicles):
            vehicle_to_customers[idx] = []
            vehicle_to_capacity[idx] = self.vehicle_capacity
            c = ordered_unserved[0]
            cap = self.vehicle_capacity
            #while the car can still serve the next customer
            while (c is not None and len(ordered_unserved) > 0 and self.demandOfCustomer[c] < cap):
                unserved_customers.remove(c)
                ordered_unserved.remove(c)
                vehicle_to_customers[idx].append(c)
                vehicle_to_capacity[idx] -= self.demandOfCustomer[c]
                cap -= self.demandOfCustomer[c]
                #assuming these leftover customers have  large demand, we try to find the smallest customer it can serve
                c = self.get_smallest_customer(ordered_unserved,c,cap)
                
            idx += 1
        print(ordered_unserved)
        for el in ordered_unserved:
            print(self.demandOfCustomer[el])
        return vehicle_to_customers, self.num_vehicles, vehicle_to_capacity, unserved_customers


