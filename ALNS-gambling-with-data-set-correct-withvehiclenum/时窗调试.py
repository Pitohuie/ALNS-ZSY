def nearest_neighbor_solution(instance, vehicle_type, customers):
    routes = []
    battery_capacity = instance.B_star  # 电动车初始电量为满电
    instance.calculate_energy_consumption_factors()  # 计算能耗系数
    charging_stations = set(instance.type_to_indices['f'])  # 使用类型字典获取充电站的索引
    depot_index = instance.type_to_indices['d'][0]  # 获取配送中心索引

    while customers:
        current_route = [depot_index]  # 路线从配送中心开始
        remaining_capacity = instance.Q_e if vehicle_type == 'electric' else instance.Q_f
        remaining_battery = battery_capacity if vehicle_type == 'electric' else None
        current_location = depot_index
        charged_once = False  # 标记是否已经充过电（仅对电动车）
        total_energy_consumed = 0  # 初始化总电量消耗
        current_time = 0  # 初始化当前时间（假设从0时刻开始）

        while customers:
            nearest_customer = None
            nearest_distance = float('inf')
            earliest_ready_time = float('inf')

            for customer in customers:
                distance = instance.d_ij[current_location, customer]
                travel_time = instance.t_ijk_e[current_location, customer] if vehicle_type == 'electric' else instance.t_ijk_f[current_location, customer]
                arrival_time = current_time + travel_time

                if arrival_time < instance.E_i[customer - 22]:  # 如果到达时间早于Ready Time
                    waiting_time = instance.E_i[customer - 22] - arrival_time
                    arrival_time = instance.E_i[customer - 22]  # 等待直到服务时间开始
                else:
                    waiting_time = 0

                if distance < nearest_distance and instance.q_i[customer - 22] <= remaining_capacity:
                    nearest_customer = customer
                    nearest_distance = distance
                    earliest_ready_time = arrival_time

            if nearest_customer is None:
                break

            # 打印当前到达时间和等待时间
            print(f"Vehicle {vehicle_type} arrives at customer {nearest_customer} at time {arrival_time:.2f}, waiting for {waiting_time:.2f} time units.")

            # 更新当前时间
            current_time = earliest_ready_time + instance.locations[nearest_customer].service_time

            # 电动车需要检查电量（省略电池相关部分的代码）
            # ...

            # 添加客户到当前路径
            current_route.append(nearest_customer)
            customers.remove(nearest_customer)

            # 更新当前位置
            current_location = nearest_customer

        # 添加返回配送中心的逻辑（省略）
        # ...

        routes.append((current_route, total_energy_consumed))

    return routes
