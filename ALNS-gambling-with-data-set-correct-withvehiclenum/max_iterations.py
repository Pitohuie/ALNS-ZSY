from alns.State import State
from numpy.random import RandomState
from alns.stop import StoppingCriterion

class MaxIterations(StoppingCriterion):
    """
    Stops the algorithm after a fixed number of iterations.
    """

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.iteration = 0

    def __call__(self, rnd: RandomState, best: State, current: State) -> bool:
        self.iteration += 1
        return self.iteration >= self.max_iterations
