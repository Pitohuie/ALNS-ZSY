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
                        'id': parts[0],  # 将id处理为字符串类型
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

# 测试函数
if __name__ == "__main__":
    file_path = "c101_21.txt"  # 替换为你的文件路径
    locations_data, vehicles_data = read_solomon_instance(file_path)
    print("Locations:", locations_data)
    print("Vehicles:", vehicles_data)
