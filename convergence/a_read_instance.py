def read_solomon_instance(file_path):
    locations = []
    vehicles = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 0:
                continue
            if parts[0] == 'StringID':
                continue
            if line.startswith('Q Vehicle fuel tank capacity'):
                vehicles['fuel_tank_capacity'] = float(parts[-1].strip('/'))
            elif line.startswith('C Vehicle load capacity'):
                vehicles['load_capacity'] = float(parts[-1].strip('/'))
            elif line.startswith('r fuel consumption rate'):
                vehicles['fuel_consumption_rate'] = float(parts[-1].strip('/'))
            elif line.startswith('g inverse refueling rate'):
                vehicles['inverse_refueling_rate'] = float(parts[-1].strip('/'))
            elif line.startswith('v average Velocity'):
                vehicles['average_velocity'] = float(parts[-1].strip('/'))
            else:
                if len(parts) == 8:
                    location = {
                        'id': parts[0],  # Keep 'id' as a string
                        'type': parts[1],
                        'x': float(parts[2]),
                        'y': float(parts[3]),
                        'demand': float(parts[4]),
                        'ready_time': float(parts[5]),
                        'due_date': float(parts[6]),
                        'service_time': float(parts[7])
                    }
                    locations.append(location)

    return locations, vehicles


# Testing the function
if __name__ == "__main__":
    test_file_path = "c101_21.txt"  # Replace with your actual file path
    locations_data, vehicles_data = read_solomon_instance(test_file_path)
    print("Locations:", locations_data)
    print("Vehicles:", vehicles_data)
