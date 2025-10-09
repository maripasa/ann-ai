module Search where

import Data.Foldable
import qualified Data.Set as Set

-- | Generic Problem
data Problem s a = Problem
  { successor :: s -> a -> (s, Int),
    isGoal :: s -> Bool,
    initial :: s,
    actions :: s -> [a]
  }

-- | Search Strategy
newtype Strategy s a = Strategy ([(s, [a], Int)] -> ((s, [a], Int), [(s, [a], Int)]))

-- | Generic Tree Search
treeSearch ::
  Problem s a -> 
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
  Problem s a -> -- Problem
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

-- | Graph Problem
data Graph a = Graph
  { edges :: [(a, [(a, Int)])], -- Node ID -> list of neighbor IDs
    start :: a,
    goal :: a
  }

graphProblem :: (Eq a) => Graph a -> Problem a a
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
        Nothing -> error "Node not present in the graph"
