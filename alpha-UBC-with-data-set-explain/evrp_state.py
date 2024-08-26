from alns.State import State  # 确保导入 State 类

class EVRPState(State):
    def __init__(self, solution, instance):
        """
        初始化 EVRPState 类的实例。

        参数:
        solution (dict): 当前的解决方案，包含常规车辆和电动车的路线。
        instance (CustomEVRPInstance): 问题实例，包含距离矩阵和各种成本参数。
        """
        self.solution = solution  # 保存当前的解决方案
        self.instance = instance  # 保存问题实例

    def objective(self):
        """
        计算当前解决方案的目标值。

        返回:
        float: 当前解决方案的总成本。
        """
        return calculate_objective(self.solution)  # 调用 calculate_objective 函数计算目标值

    def copy(self):
        """
        创建当前状态的副本。

        返回:
        EVRPState: 当前状态的副本。
        """
        return EVRPState(self.solution.copy(), self.instance)  # 创建解决方案的副本并返回新的状态实例


def calculate_objective(solution):
    """
    计算给定解决方案的总成本。

    参数:
    solution (dict): 包含常规车辆和电动车路线的解决方案。

    返回:
    float: 总路线成本。
    """
    route_cost = 0  # 初始化总路线成本为 0

    # 计算常规车辆的总路线成本
    for route in solution["conventional"]:
        for i in range(len(route) - 1):
            route_cost += solution["instance"].distance_matrix[route[i], route[i + 1]] * solution["instance"].p_3
            # 路线成本 = 距离矩阵中的距离 * 常规车辆的单位运输成本

    # 计算电动车的总路线成本
    for route in solution["electric"]:
        for i in range(len(route) - 1):
            route_cost += solution["instance"].distance_matrix[route[i], route[i + 1]] * solution["instance"].p_3
            # 路线成本 = 距离矩阵中的距离 * 电动车的单位运输成本

    return route_cost  # 返回总路线成本
