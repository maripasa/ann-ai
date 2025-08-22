import copy

class State:
    pass

class Problem:
    initial_state = None
    # Checks if state is a goal
    def is_goal(self, state):
        raise NotImplementedError

    # Generates the next state based on an action
    def precessor(self, state, action):
        raise NotImplementedError

    # Generates all possible actions
    def expand(self, state) -> list[State]:
        raise NotImplementedError

    # Generates the path of action/states that generated this state
    def solution(self, state) -> list:
        raise NotImplementedError

class Strategy:
    # The edge of our knowledge
    def __init__(self):
        self.frontier = []

    # We exhausted the tree?
    def is_exhausted(self):
        return self.frontier == []

    # Add new nodes to our knowledge
    def add(self, node):
        self.frontier.append(node)

    # Choose a node to inspect  
    def choose(self):
        raise NotImplementedError

class Bfs(Strategy):
    def choose(self):
        return self.frontier.pop(0)

class Dfs(Strategy):
    def choose(self):
        return self.frontier.pop()

class PacmanState(State):
    def __init__(self, table, position, valid_actions, parent):
        self.table = table
        self.position = position
        self.valid_actions = valid_actions
        # (Action, Parent State)
        self.parent = parent

class Pacman(Problem):
    directions = {
            "left": (0, -1),
            "up": (-1, 0),
            "right": (0, 1),
            "down": (1, 0),
            }

    def __init__(self, table: list[list]):
        position = (-1, -1)

        for i in range(len(table)):
            for j in range(len(table[0])):
                if table[i][j] == 'c':
                    position = (i,j)
                    break

        if position == (-1,-1):
            raise ValueError("No Pacman")
        
        valid_actions = self.valid_actions(position, table)

        self.initial_state = PacmanState(table, position, valid_actions, (None, None))

    def is_goal(self, state: PacmanState): # type: ignore
        return all(item != "p" for line in state.table for item in line)

    def precessor(self, state: PacmanState, action: str): # type: ignore
        if action not in state.valid_actions:
            raise ValueError("Invalid Action")

        dy, dx = self.directions[action]
        old = state.position
        new = (old[0] + dy, old[1] + dx)
        table = copy.deepcopy(state.table)
        table[new[0]][new[1]] = "c"
        table[old[0]][old[1]] = "v"

        valid_actions = self.valid_actions(new, table)

        return PacmanState(table, position=new, valid_actions=valid_actions, parent=(action, state))

    def expand(self, state: PacmanState): # type: ignore
        return [self.precessor(state, action) for action in state.valid_actions]
        
    def valid_actions(self, position, table):
        valid_actions = []

        for action in self.directions.keys():
            dy, dx = self.directions[action]
            new = (position[0] + dy, position[1] + dx)
            if new[0] < 0 or new[0] >= len(table):
                continue
            if new[1] < 0 or new[1] >= len(table[0]):
                continue
            if table[new[0]][new[1]] == "w":
                continue
            valid_actions.append(action)

        return valid_actions

    def solution(self, state: PacmanState):
        actions = []
        current = state
        while current.parent != (None, None):
            action, parent = current.parent
            actions.append(action)
            current = parent
        actions.reverse()
        return actions

def tree_search(problem: Problem, strategy: Strategy):
    strategy.add(problem.initial_state)
    while True:
        print(len(strategy.frontier))
        if strategy.is_exhausted():
            return False
        choice = strategy.choose()
        if problem.is_goal(choice):
            return problem.solution(choice)
        for state in problem.expand(choice):
            strategy.add(state)

pac = Pacman([
             ['p', 'p', 'p', 'p'],
             ['p', 'v', 'v', 'v'],
             ['p', 'v', 'v', 'v'],
             ['p', 'v', 'v', 'c']
            ])

strat = Dfs()



def interactive_tree_search_debug(problem: Problem, strategy: Strategy):
    strategy.add(problem.initial_state)
    while True:
        if strategy.is_exhausted():
            print("Frontier esgotada, não encontrou solução")
            return False

        # Determinar o estado com menos pontos ainda não comidos
        min_pontos = min(
            sum(row.count('p') for row in s.table)
            for s in strategy.frontier
        )
        estado_promissor = [
            s for s in strategy.frontier 
            if sum(row.count('p') for row in s.table) == min_pontos
        ][0]

        # Imprimir estados na fronteira
        print("\n=== Frontier ===")
        for s in strategy.frontier:
            pontos = sum(row.count('p') for row in s.table)
            print(f"Pos: {s.position}, Ações válidas: {s.valid_actions}, Pontos restantes: {pontos}")
            for row in s.table:
                print(" ".join(row))
            print("---")

        # Destacar estado mais promissor
        pontos = sum(row.count('p') for row in estado_promissor.table)
        print(f">>> Estado mais promissor: Pos {estado_promissor.position}, Pontos restantes: {pontos}")

        input("Pressione Enter para avançar...")

        choice = strategy.choose()
        print(f"\n>>> Escolhido: Pos {choice.position}, Ações válidas: {choice.valid_actions}")
        for row in choice.table:
            print(" ".join(row))

        if problem.is_goal(choice):
            print("Objetivo alcançado!")
            return problem.solution(choice)

        for state in problem.expand(choice):
            strategy.add(state)

interactive_tree_search_debug(pac, strat)
