import numpy as np

def read_solomon_instance(file_path):
    locations = []
    vehicles = {}

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('StringID'):
                    continue
                if line.startswith('Q Vehicle fuel tank capacity'):
                    vehicles['fuel_tank_capacity'] = float(line.split('/')[-2])
                elif line.startswith('C Vehicle load capacity'):
                    vehicles['load_capacity'] = float(line.split('/')[-2])
                elif line.startswith('r fuel consumption rate'):
                    vehicles['fuel_consumption_rate'] = float(line.split('/')[-2])
                elif line.startswith('g inverse refueling rate'):
                    vehicles['inverse_refueling_rate'] = float(line.split('/')[-2])
                elif line.startswith('v average Velocity'):
                    vehicles['average_velocity'] = float(line.split('/')[-2])
                else:
                    parts = line.split()
                    if len(parts) == 8:
                        try:
                            location = {
                                'id': parts[0],
                                'type': parts[1],
                                'x': float(parts[2]),
                                'y': float(parts[3]),
                                'demand': float(parts[4]),
                                'ready_time': float(parts[5]),
                                'due_date': float(parts[6]),
                                'service_time': float(parts[7])
                            }
                            locations.append(location)
                        except ValueError as e:
                            print(f"Error parsing line: {line}. Error: {e}")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return locations, vehicles
