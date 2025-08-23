from collections import deque

class Strategy:
    # The edge of our knowledge
    def __init__(self):
        self.frontier = deque()

    # We exhausted the tree?
    def is_exhausted(self):
        return not self.frontier

    # Add new nodes to our knowledge
    def add(self, node):
        self.frontier.append(node)

    # Choose a node to inspect  
    def choose(self):
        raise NotImplementedError

class Bfs(Strategy):
    def choose(self):
        return self.frontier.popleft()

class Dfs(Strategy):
    def choose(self):
        return self.frontier.pop()
