# TODO: Add smth that stops repeating states

class State:
    pass

class Problem:
    initial_state = None
    # Checks if state is a goal
    def is_goal(self, state):
        raise NotImplementedError

    # Generates the next state based on an action
    def successor(self, state, action):
        raise NotImplementedError

    # Generates all possible actions
    def expand(self, state) -> list:
        raise NotImplementedError

    # Generates the path of action/states that generated this state
    def solution(self, state) -> list:
        raise NotImplementedError

class PacmanState(State):
    def __init__(self, pellets, walls, position, board_size, parent_state, parent_action):
        self.pellets = pellets # set
        self.position = position # 2tuple
        self.walls = walls # set
        self.board_size = board_size
        self.parent_state = parent_state # PacmanState (mostly for debug)
        self.parent_action = parent_action # str

    def __str__(self):
            rows, cols = self.board_size
            board = [["v" for _ in range(cols)] for _ in range(rows)]

            for y, x in self.walls:
                board[y][x] = "w"
            for y, x in self.pellets:
                board[y][x] = "p"

            py, px = self.position
            board[py][px] = "c"

            return "\n".join("".join(row) for row in board)

class Pacman(Problem):
    """
    If we consider y as up and down and x as left and right we have every direction as (y, x)
    0,0 is the leftmost highest square

    """
    DIRECTIONS = [
            ("up", (-1, 0)),
            ("down", (1, 0)),
            ("left", (0, -1)),
            ("right", (0, 1)),
            ]

    def __init__(self, table: list[list] | PacmanState):
        if isinstance(table, PacmanState):
            self.initial_state = table
            return

        pellets = []
        walls = []
        position = None
        board_size = (len(table), len(table[0]))

        for i in range(len(table)):
            for j in range(len(table[0])):
                match table[i][j]:
                    case "c":
                        if position is not None:
                            raise ValueError("More than one pacman")
                        position = (i, j)
                    case "p":
                        pellets.append((i, j))
                    case "w":
                        walls.append((i, j))

        if position is None:
            raise ValueError("No Pacman")
        
        self.initial_state = PacmanState(set(pellets), set(walls), position, board_size, None, None)


    def is_goal(self, state: PacmanState): # type: ignore
        return not state.pellets

    def successor(self, state: PacmanState, action): # type: ignore
        name, new_position = action
        new_pellets = set(state.pellets)
        new_pellets.discard(new_position)

        return PacmanState(new_pellets, state.walls, new_position, state.board_size, state, name)

    def expand(self, state: PacmanState): # type: ignore
        valid_actions = self.valid_actions(state)
        return [self.successor(state, action) for action in valid_actions]
        
    def valid_actions(self, state):
        valid_actions = []
        oy, ox = state.position
        r, c = state.board_size

        for action, (dy, dx) in self.DIRECTIONS:
            ny, nx = oy + dy, ox + dx
            if 0 <= ny < r and 0 <= nx < c and (ny, nx) not in state.walls:
                valid_actions.append((action, (ny, nx)))

        return valid_actions

    def solution(self, state: PacmanState):
        actions = []
        current = state
        while current.parent_state is not None:
            parent = current.parent_state
            actions.append(current.parent_action)
            current = parent
        actions.reverse()
        return actions
