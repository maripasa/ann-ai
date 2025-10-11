module Search where

import Data.Foldable
import Data.List
import Data.Ord
import qualified Data.Set as Set

-- | Generic Problem
data Problem s a c = Problem
  { successor :: s -> a -> (s, c),
    isGoal :: s -> Bool,
    initial :: s,
    actions :: s -> [a]
  }

-- | Search Strategy
type Strategy s a c = ([(s, [a], c)] -> ((s, [a], c), [(s, [a], c)]))

-- | Generic Tree Search
treeSearch :: (Num c) => Problem s a c -> Strategy s a c -> Maybe [a]
treeSearch problem = search [(initial problem, [], 0)] problem
  where
    search [] _ _ = Nothing
    search frontier problem' strategy'
      | isGoal problem' node = Just path
      | otherwise = search (frontier' ++ expandedNodes) problem' strategy'
      where
        ((node, path, cost), frontier') = strategy' frontier
        expandedNodes =
          [ (s', path ++ [act], cost + stepCost) | act <- actions problem' node, let (s', stepCost) = successor problem' node act
          ]

-- | Generic Graph Search
graphSearch :: (Ord s, Num c) => Problem s a c -> Strategy s a c -> Maybe [a]
graphSearch problem = search Set.empty [(initial problem, [], 0)] problem
  where
    search _ [] _ _ = Nothing
    search closed frontier problem' strategy'
      | isGoal problem' node = Just path
      | Set.member node closed = search closed frontier' problem' strategy'
      | otherwise =
          search
            (Set.insert node closed)
            (frontier' ++ expandedNodes)
            problem'
            strategy'
      where
        ((node, path, cost), frontier') = strategy' frontier
        expandedNodes =
          [ (s', path ++ [act], cost + stepCost) | act <- actions problem' node, let (s', stepCost) = successor problem' node act
          ]

-- May fail if filtered = []
depthLimitedSearch :: Int -> Strategy s a c
depthLimitedSearch limit = depthFirstSearch . takeWhile (\(_, a, _) -> length a <= limit)

-- | max' is for preventing infinite search. You could set it as -1 for infinite search.
iterativeDeepeningSearch :: Int -> Problem s a c -> (Problem s a c -> Strategy s a c -> Maybe [a]) -> Maybe [a]
iterativeDeepeningSearch max' problem search = go 0
  where
    go limit
      | limit == max' = Nothing
      | otherwise = case search problem (depthLimitedSearch limit) of
          Just path -> Just path
          Nothing -> go (limit + 1)

depthFirstSearch :: Strategy s a c
depthFirstSearch frontier = (last frontier, init frontier)

breadthFirstSearch :: Strategy s a c
breadthFirstSearch [] = error "Breadth First Search can't use an empty frontier"
breadthFirstSearch (state : fs) = (state, fs)

uniformCostSearch :: (Ord c) => Strategy s a c
uniformCostSearch frontier =
  let sorted = sortBy (comparing (\(_, _, cost) -> cost)) frontier
   in (head sorted, tail sorted)

greedySearch :: (Ord c) => (s -> c) -> Strategy s a c
greedySearch h frontier =
  let sorted = sortBy (comparing (\(state, _, _) -> h state)) frontier
   in (head sorted, tail sorted)

aStar :: (Num c, Ord c) => (s -> c) -> Strategy s a c
aStar h frontier =
  let sorted = sortBy (comparing (\(state, _, cost) -> cost + h state)) frontier
   in (head sorted, tail sorted)

-- | Generic Hill Climb
hillClimb :: (Ord v) => Problem s a c -> (s -> v) -> s
hillClimb problem valuator = climb (initial problem)
  where
    climb state
      | null successors || valuator contender <= valuator state = state
      | otherwise = climb contender
      where
        contender = maximumBy (comparing valuator) successors
        successors = map (fst . successor problem state) . actions problem $ state

-- | Graph Problem
data Graph a c = Graph
  { edges :: [(a, [(a, c)])], -- Node ID -> list of neighbor IDs
    start :: a,
    goal :: a
  }

graphProblem :: (Eq a) => Graph a c -> Problem a a c
graphProblem graph =
  Problem
    { successor = successor' graph,
      isGoal = (== goal graph),
      initial = start graph,
      actions = map fst . snd . findByNode (edges graph)
    }
  where
    successor' graph' node target =
      (`findByNode` target) . snd . findByNode (edges graph') $ node
    findByNode space node =
      case find (\(n, _) -> n == node) space of
        Just a -> a
        Nothing -> error "Node being choosed is not present in the graph"
