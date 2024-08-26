import copy
from b_CCMFEVRP_PRTW_instance import CustomEVRPInstance


class CcmfevrpState:
    """
    CCMFEVRP-PRTW 的解状态类。
    """

    def __init__(self, routes, unassigned=None):
        """
        初始化状态，routes 是路线列表，每个子列表表示一辆车的路线。
        unassigned 表示未分配的客户列表，默认为空列表。
        """
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        """
        深拷贝当前状态，返回一个新副本。
        """
        return CcmfevrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self, instance: CustomEVRPInstance):
        """
        计算总路线成本，考虑电动车和燃油车的不同类型。
        """
        total_cost = 0
        for route in self.routes:
            if self.is_electric_vehicle_route(route, instance):
                total_cost += self.route_cost(route, instance, vehicle_type='electric')
            else:
                total_cost += self.route_cost(route, instance, vehicle_type='fuel')
        return total_cost

    @property
    def cost(self, instance: CustomEVRPInstance):
        """
        目标函数的别名，用于绘图或快速成本评估。
        """
        return self.objective(instance)

    def find_route(self, customer):
        """
        返回包含指定客户的路线。
        如果未找到，抛出 ValueError 异常。
        """
        for route in self.routes:
            if customer in route:
                return route
        raise ValueError(f"解中未包含客户 {customer}。")

    def route_cost(self, route, instance: CustomEVRPInstance, vehicle_type='electric'):
        """
        根据车辆类型计算给定路线的成本。
        """
        distances = instance.d_ij
        tour = [instance.location_id_to_index[instance.O.id]] + route + [
            instance.location_id_to_index[instance.O_prime]]

        if vehicle_type == 'electric':
            return sum(distances[tour[idx]][tour[idx + 1]] / instance.v_e for idx in range(len(tour) - 1))
        else:  # 燃油车的成本计算
            return sum(distances[tour[idx]][tour[idx + 1]] / instance.v_f for idx in range(len(tour) - 1))

    def is_electric_vehicle_route(self, route, instance: CustomEVRPInstance):
        """
        判断当前路线是否由电动车执行。
        可以根据具体属性或规则来确定。
        """
        # 示例规则：索引小于 instance.K_e 的路线是电动车路线
        route_index = self.routes.index(route)
        return route_index < instance.K_e
