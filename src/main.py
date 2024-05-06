import sys
import os
from pathlib import Path
from model_timer import Timer   # You need to implement a timer module or use an existing one
from vrp_parser import VRPInstance  # Assuming VRPInstance is defined in a separate module
from solver import begin_search
from solver import VRPState

def main():
    if len(sys.argv) == 1:
        print("Usage: python main.py <file>")
        return

    input_file = sys.argv[1]
    file_name = Path(input_file).name
    print("Instance:", input_file)

    watch = Timer()
    watch.start()
    instance = VRPInstance(input_file)
    res : VRPState = begin_search(vrp_instance=instance)
    watch.stop()

    
    sol = ""
    #writing solution in the correct format
    for car, route in res.vehicle_to_route.items():
        if car == len(res.vehicle_to_route)-1: #the last car
            continue
        for i in range(len(route)):
            customer = route[i]
            if i == 0:
                sol += "0 "
                sol += str(customer) + " "
            elif i == len(route) -1:
                sol += str(customer) + " "
                sol += "0 "
            else:
                sol += str(customer) + " "
        if len(route) == 0:
            sol += "0 0 "

    print("{\"Instance\": \"" + file_name +
          "\", \"Time\": " + "{:.2f}".format(watch.get_elapsed()) +
          ", \"Result\": " + "{:.2f}".format(res.objective()) +
          ", \"Solution\": " + sol + "}")
    
    new_file_name = "solutions/" + file_name + ".sol"
    if os.path.exists(new_file_name):
        os.remove(new_file_name)
    with open(new_file_name, "w") as file:
        file.write("{:.2f}".format(res.objective()))
        file.write("\n")
        for car, route in res.vehicle_to_route.items():
            if car == res.num_vehicles:
                continue
            sol = ""
            for i in range(len(route)):
                customer = route[i]
                if i == 0:
                    sol += "0 "
                    sol += str(customer) + " "
                elif i == len(route) -1:
                    sol += str(customer) + " "
                    sol += "0 "
                else:
                    sol += str(customer) + " "
            if len(route) == 0:
                sol+= "0"
            sol += "\n"
            file.write(sol)

if __name__ == "__main__":
    main()
