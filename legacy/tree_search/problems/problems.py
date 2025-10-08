# TODO: Add smth that stops repeating states
"""state, cost = successor(state, action)"""
    
class State:
    pass

class Problem:
    def __init__(self):
        self.initial_state = None

    # Checks if state is a goal
    def is_goal(self, state):
        raise NotImplementedError

    # Generates the path of action/states that generated this state
    def solution(self, state) -> list:
        raise NotImplementedError

    # Says what actions are possible next
    def valid_actions(self, state) -> list:
        raise NotImplementedError

    # Generates the next state based on an action
    def successor(self, state, action):
        raise NotImplementedError

    # Generates all possible actions
    def expand(self, state): # type: ignore
        valid_actions = self.valid_actions(state)
        return [self.successor(state, action) for action in valid_actions]
