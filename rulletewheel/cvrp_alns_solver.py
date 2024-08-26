%matplotlib
inline

import copy
from types import SimpleNamespace

import vrplib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from alns import ALNS
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

# 设置随机种子
SEED = 1234
rnd.seed(SEED)

# 示例数据
data = {
    "edge_weight": np.random.randint(1, 100, size=(12, 12)),  # 示例距离矩阵
    "node_coord": {i: (rnd.randint(0, 100), rnd.randint(0, 100)) for i in range(12)}  # 随机生成节点坐标
}


class CvrpState:
    """
    CVRP的解状态。它有两个数据成员，routes和unassigned。
    Routes是一个整数列表的列表，每个内部列表对应一条路线，表示要访问的客户的顺序。
    路线不包含起点和终点。Unassigned是一个整数列表，每个整数表示一个未分配的客户。
    """

    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        return CvrpState(copy.deepcopy(self.routes), self.unassigned.copy())

    def objective(self):
        """
        计算总路线成本。
        """
        return sum(route_cost(route) for route in self.routes)

    @property
    def cost(self):
        """
        objective方法的别名。用于绘图。
        """
        return self.objective()

    def find_route(self, customer):
        """
        返回包含传入客户的路线。
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")


def route_cost(route):
    distances = data["edge_weight"]
    tour = [0] + route + [0]

    return sum(distances[tour[idx]][tour[idx + 1]]
               for idx in range(len(tour) - 1))


# 绘制解决方案函数
def plot_solution(solution, name="CVRP solution"):
    """
    Plot the routes of the passed-in solution.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))

    for idx, route in enumerate(solution.routes):
        ax.plot(
            [data["node_coord"][loc][0] for loc in [0] + route + [0]],
            [data["node_coord"][loc][1] for loc in [0] + route + [0]],
            color=cmap[idx],
            marker='.'
        )

    # Plot the depot
    kwargs = dict(label="Depot", zorder=3, marker="*", s=750)
    ax.scatter(*data["node_coord"][0], c="tab:red", **kwargs)

    ax.set_title(f"{name}\n Total distance: {solution.cost}")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend(frameon=False, ncol=3)
    plt.show()


# 示例使用
initial_routes = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
unassigned_customers = [10, 11]

cvrp_state = CvrpState(initial_routes, unassigned_customers)

# 打印初始状态的目标值
print("Initial objective value:", cvrp_state.objective())

# 复制状态并修改
cvrp_state_copy = cvrp_state.copy()
cvrp_state_copy.routes[0].append(10)

# 打印修改后的目标值
print("Modified objective value:", cvrp_state_copy.objective())

# ALNS实现
alns = ALNS(rnd.RandomState(SEED))


# 定义破坏和修复操作
# 示例：
def random_removal(state):
    state = state.copy()
    route_idx = rnd.randint(len(state.routes))
    customer_idx = rnd.randint(len(state.routes[route_idx]))
    customer = state.routes[route_idx].pop(customer_idx)
    state.unassigned.append(customer)
    return state


def random_repair(state):
    state = state.copy()
    if not state.unassigned:
        return state
    customer = state.unassigned.pop(rnd.randint(len(state.unassigned)))
    route_idx = rnd.randint(len(state.routes))
    state.routes[route_idx].append(customer)
    return state


alns.add_destroy_operator(random_removal)
alns.add_repair_operator(random_repair)

# 设置接受标准、选择机制和停止标准
criterion = RecordToRecordTravel(0.01, 1)
selector = RouletteWheel()
stopping = MaxRuntime(60)  # 运行时间60秒

# 求解问题
result = alns.iterate(cvrp_state, criterion, selector, stopping)

# 绘制结果
fig, ax = plt.subplots()
ax.plot(result.objectives, label='Objective Value')
ax.set_xlabel('Iteration')
ax.set_ylabel('Objective Value')
ax.legend()
plt.show()

# 打印最佳解
print("Best solution found has objective value:", result.best_state.objective())
print("Best routes:", result.best_state.routes)

# 绘制最佳解
plot_solution(result.best_state)
