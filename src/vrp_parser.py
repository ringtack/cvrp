
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

