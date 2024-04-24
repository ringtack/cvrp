import vrp_parser as vp

# Main function
def main():
    # Load the data
    data = vp.parse_input("5_4_10.vrp")

    # Print the data
    print(data)

if __name__ == "__main__":
    main()