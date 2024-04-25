import sys
import os
from pathlib import Path
from timer import Timer  # You need to implement a timer module or use an existing one
from vrp_parser import VRPInstance  # Assuming VRPInstance is defined in a separate module
from solver import Solver  # Assuming Solver is defined in a separate module

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
    solver = Solver(instance)
    solution = solver.main()
    watch.stop()

    sol = ""
    #writing solution in the correct format

    print("{\"Instance\": \"" + file_name +
          "\", \"Time\": " + "{:.2f}".format(watch.get_time()) +
          ", \"Result\": " + "{:.2f}".format(solver.objective_value) +
          ", \"Solution\": " + sol + "}")

if __name__ == "__main__":
    main()
