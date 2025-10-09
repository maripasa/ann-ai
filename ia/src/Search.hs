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
newtype Strategy s a = Strategy ([(s, [a], Int)] -> ((s, [a], Int), [(s, [a], Int)]))

-- | Generic Tree Search
treeSearch ::
  Problem s a c-> 
  Strategy s a -> 
  Maybe [a]
treeSearch problem (Strategy strategy) =
  search [(initial problem, [], 0)] problem strategy
  where
    search [] _ _ = Nothing
    search fringe problem' strategy'
      | isGoal problem' node = Just path
      | otherwise = search (fringe' ++ expandedNodes) problem' strategy'
      where
        ((node, path, cost), fringe') = strategy' fringe
        expandedNodes =
          [ (s', path ++ [act], cost + stepCost) | act <- actions problem' node,
            let (s', stepCost) = successor problem' node act
          ]

-- | Generic Graph Search
graphSearch ::
  (Ord s) =>
  Problem s a c -> -- Problem
  Strategy s a -> -- Strategy
  Maybe [a] -- Solution
graphSearch problem (Strategy strategy) =
  search Set.empty [(initial problem, [], 0)] problem strategy
  where
    search ::
      (Ord s) =>
      Set.Set s -> -- Closed
      [(s, [a], Int)] -> -- Fringe
      Problem s a ->
      ([(s, [a], Int)] -> ((s, [a], Int), [(s, [a], Int)])) ->
      Maybe [a]
    search _ [] _ _ = Nothing
    search closed fringe problem' strategy'
      | isGoal problem' node   = Just path
      | Set.member node closed = search closed fringe' problem' strategy'
      | otherwise =
          search (Set.insert node closed)
                 (fringe' ++ expandedNodes)
                 problem'
                 strategy'
      where
        ((node, path, cost), fringe') = strategy' fringe
        expandedNodes =
          [ (s', path ++ [act], cost + stepCost) | act <- actions problem' node
          , let (s', stepCost) = successor problem' node act
          ]

-- Uninformed Search
depthFirstSearch :: Strategy s a
depthFirstSearch = Strategy (\fringe -> (last fringe, init fringe))

breadthFirstSearch :: Strategy s a
breadthFirstSearch = Strategy bfs
  where
    bfs [] = error "Breadth First Search can't use an empty fringe"
    bfs (state:fs) = (state, fs)

uniformCostSearch :: Strategy s a
uniformCostSearch = Strategy $ \fringe ->
  let sorted = sortBy (comparing (\(_,_,cost) -> cost)) fringe
  in (head sorted, tail sorted)

-- Informed Search
greedySearch :: (Ord a) => (s -> a) -> Strategy s a
greedySearch h = Strategy $ \fringe ->
  let sorted = sortBy (comparing (\(state,_,_) -> h state)) fringe
  in (head sorted, tail sorted)

aStar :: (Num b, Ord b) => (s -> b) -> Strategy s a
aStar h = Strategy $ \fringe ->
  let sorted = sortBy (comparing (\(state, _, cost) -> cost + h state)) fringe
  in (head sorted, tail sorted)

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
        Nothing -> error "findByNode -> Node being choosed is not present in the graph"
