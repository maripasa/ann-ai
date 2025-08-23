from problems import Problem, Pacman
from strategies import Strategy, Bfs, Dfs

def tree_search(problem: Problem, strategy: Strategy):
    strategy.add(problem.initial_state)
    while True:
        if strategy.is_exhausted():
            return []
        print(len(strategy.frontier))
        choice = strategy.choose()
        if problem.is_goal(choice):
            return problem.solution(choice)
        for state in problem.expand(choice):
            strategy.add(state)

pac = Pacman([
             ['p', 'p', 'p'],
             ['p', 'v', 'v'],
             ['p', 'v', 'c']
            ])

strat = Bfs()

print(tree_search(pac, strat))

