import math

class VRPInstance:
    def __init__(self, fileName):
        try:
            with open(fileName, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError as e:
            print("Error: in VRPInstance() " + fileName + "\n" + str(e))
            exit(-1)

        # Parsing input parameters
        self.num_customers = int(lines[0])
        self.num_vehicles = int(lines[1])
        self.vehicle_capacity = int(lines[2])

        print("Number of customers:", self.num_customers)
        print("Number of vehicles:", self.num_vehicles)
        print("Vehicle capacity:", self.vehicle_capacity)

        self.demandOfCustomer = []
        self.xCoordOfCustomer = []
        self.yCoordOfCustomer = []

        for line in lines[3:]:
            data = line.split()
            self.demandOfCustomer.append(int(data[0]))
            self.xCoordOfCustomer.append(float(data[1]))
            self.yCoordOfCustomer.append(float(data[2]))

        for i in range(self.numCustomers):
            print(self.demandOfCustomer[i], self.xCoordOfCustomer[i], self.yCoordOfCustomer[i])


    def construct_intial_solution(self):
      unserved_customers = set()
      for i in range(self.customers):
          unserved_customers.add(i)

      def find_closest_customer(self, vehicle_x, vehicle_y, unserved_customers, capacity):
          closest_idx = -1
          closest_distance = float.max('inf')
          for customer in unserved_customers:
              distance = math.sqrt((vehicle_x - self.customer_x[customer])**2 + (vehicle_y - self.customer_x[customer])**2)
              if self.customer_demand[customer] <= capacity and distance < closest_distance:
                  closest_distance = distance
                  closest_distance = customer
          return closest_idx
      
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
              customer_idx = find_closest_customer(self, x_pos, y_pos, unserved_customers, self.vehicle_to_capacity[vehicle_num])
              #if this vehicle cannot serve *any* customer it means that it has no spare capacity
              if (customer_idx == -1):
                  vehicle_num += 1
          unserved_customers.remove(customer_idx)
          self.vehicle_to_capacity[vehicle_num] -= self.customer_demand[customer_idx]
          self.vehicle_to_customers[vehicle_num].apend(customer_idx)
      return self.vehicle_to_customers, self.num_vehicles


