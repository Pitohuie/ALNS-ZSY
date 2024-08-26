from alns.State import State  # 导入 ALNS 库中的 State 类
from numpy.random import RandomState  # 导入 numpy 库中的 RandomState 类
from alns.stop import StoppingCriterion  # 导入 ALNS 库中的 StoppingCriterion 类

class MaxIterations(StoppingCriterion):
    """
    在固定的迭代次数后停止算法。
    """

    def __init__(self, max_iterations: int):
        """
        初始化 MaxIterations 类的实例。

        参数:
        max_iterations (int): 最大迭代次数。
        """
        self.max_iterations = max_iterations  # 设置最大迭代次数
        self.iteration = 0  # 初始化当前迭代次数为 0

    def __call__(self, rnd: RandomState, best: State, current: State) -> bool:
        """
        在每次迭代时调用该方法，判断是否达到停止条件。

        参数:
        rnd (RandomState): 随机数生成器实例。
        best (State): 当前最优解。
        current (State): 当前解。

        返回:
        bool: 如果达到最大迭代次数，返回 True，否则返回 False。
        """
        self.iteration += 1  # 增加当前迭代次数
        return self.iteration >= self.max_iterations  # 如果达到最大迭代次数，返回 True，否则返回 False
