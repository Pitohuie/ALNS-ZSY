from alns.State import State  # 确保导入 State 类

class EVRPState(State):
    def __init__(self, solution, instance):
        self.solution = solution
        self.instance = instance

    def objective(self):
        return calculate_objective(self.solution)

    def copy(self):
        return EVRPState(self.solution.copy(), self.instance)


def calculate_objective(solution):
    route_cost = 0

    for route in solution["conventional"]:
        for i in range(len(route) - 1):
            route_cost += solution["instance"].distance_matrix[route[i], route[i + 1]] * solution["instance"].p_3

    for route in solution["electric"]:
        for i in range(len(route) - 1):
            route_cost += solution["instance"].distance_matrix[route[i], route[i + 1]] * solution["instance"].p_3

    return route_cost
