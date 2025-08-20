import random
from itertools import product
import operator as opr

def scalar(a: list, c: int):
    return [i * c for i in a]

def add(a: list, b: list) -> list:
    if len(a) != len(b):
        raise ValueError("not same length")
    return [a[i] + b[i] for i in range(len(a))]

def dot(a: list, b: list) -> int:
    if len(a) != len(b):
        raise ValueError("Vec not the same length")
    return sum([a[i] * b[i] for i in range(len(a))])

def train(space: list[tuple[list, int]], weights: list, turns: int) -> list:
    for _ in range(turns):
        x,y = random.choice(space)
        if (dot(x, weights) >= 0 and y == -1) or (dot(x, weights) < 0 and y == +1):
            weights = add(weights, scalar(x, y))
    return weights

def sign(x):
    return 1 if x>=0 else -1

def main():
    nums = list(product([0, 1], repeat=2))
    d = 2 
    t = 100

    ws = zip(["AND", "OR", "XOR"],
            [train(
                space   = [([-1] + list(pair), 2 * (op(pair[0],pair[1])) - 1) for pair in nums],
                weights = [0]*(d+1),
                turns   = t
                )
            for op in [opr.and_, opr.or_, opr.xor]
            ]
        )

    for op_name, w_op in ws:
        print(f"{op_name} weights: {w_op}")
        for pair in nums:
            print(f"{pair} -> {sign(dot([-1] + list(pair), w_op))}")

if __name__ == "__main__":
    main()
