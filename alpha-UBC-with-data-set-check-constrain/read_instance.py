def read_solomon_instance(file_path):
    locations = []
    vehicles = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # 去除行尾的换行符和空格
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

    return locations, vehicles
