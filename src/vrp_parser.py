import math

class VRPInstance:
    def __init__(self, fileName):
        try:
            with open(fileName, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError as e:
            print("Error: in VRPInstance() " + fileName + "\n" + str(e))
            exit(-1)
        print(lines)

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
        for line in lines[1:-1]: #last line is an empty string
            data = line.split()
            self.demandOfCustomer.append(int(data[0]))
            self.xCoordOfCustomer.append(float(data[1]))
            self.yCoordOfCustomer.append(float(data[2]))
        for i in range(self.num_customers):
            print(self.demandOfCustomer[i], self.xCoordOfCustomer[i], self.yCoordOfCustomer[i])
        

    def construct_intial_solution(self):
        unserved_customers = set()
        for i in range(1,self.num_customers):
            unserved_customers.add(i)

        def find_closest_customer(self, vehicle_x, vehicle_y, unserved_customers, capacity):
            closest_idx = -1
            closest_distance = float('inf')
            for customer in unserved_customers:
                distance = math.sqrt((vehicle_x - self.xCoordOfCustomer[customer])**2 + (vehicle_y - self.yCoordOfCustomer[customer])**2)
                if self.demandOfCustomer[customer] <= capacity and distance < closest_distance:
                    closest_distance = distance
                    closest_idx = customer
            return closest_idx
      
        vehicle_num = 0
        vehicle_to_capacity = {}
        vehicle_to_customers = {}
        for v in range(self.num_vehicles):
            vehicle_to_capacity[v] = self.vehicle_capacity
            vehicle_to_customers[v] = []

        while (len(unserved_customers) > 0):
            customer_idx = -1
            while (customer_idx == -1):
                x_pos = -1
                y_pos = -1
                #no customers visited so far
                if len(vehicle_to_customers[vehicle_num]) == 0:
                    x_pos = 0
                    y_pos = 0
                else:
                    last_customer_idx = vehicle_to_customers[vehicle_num][-1]
                    x_pos = self.xCoordOfCustomer[last_customer_idx]
                    y_pos = self.yCoordOfCustomer[last_customer_idx]
                customer_idx = find_closest_customer(self, x_pos, y_pos, unserved_customers, vehicle_to_capacity[vehicle_num])
                #if this vehicle cannot serve *any* customer it means that it has no spare capacity
                if (customer_idx == -1):
                    vehicle_num += 1
            unserved_customers.remove(customer_idx)
            vehicle_to_capacity[vehicle_num] -= self.demandOfCustomer[customer_idx]
            vehicle_to_customers[vehicle_num].append(customer_idx)
        return vehicle_to_customers, self.num_vehicles,vehicle_to_capacity


