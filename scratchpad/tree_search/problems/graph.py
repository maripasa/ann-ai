from problems.problems import Problem, State

class Graph(Problem):
    def __init__(self, initial_node, goal_node):

        self.initial_node = initial_node
        self.goal_node = goal_node

    def is_goal(self, state):
        return state is self.goal_node

    def valid_actions(self, state) -> list:
        return state.connections.keys()

    def solution(self, state):
        choices = []
        current = state
        while current.origin is not None:
            parent = current.origin
            choices.append(parent.key)
            current = parent
        choices.reverse()
        return choices

    def successor(self, state, action):
        raise NotImplementedError
