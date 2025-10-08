{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Search where

-- Generic Problem
class Action a
class State s
class (Action a, State s) => Problem p s a | p -> s a where
  successor :: p -> a -> s -> s
  isGoal :: p -> s -> Bool
  actions :: p -> s -> [a]
  startState :: p -> s
  expand :: p -> s -> [s]

-- Graph Problem
data Graph a = Graph
  { edges :: [(Node a, [Node a])], -- Node ID -> list of neighbor IDs
    start :: Node a,
    goal :: Node a
  }

newtype Node a = Node a

newtype GraphAction a = GraphAction a

instance Action (GraphAction a)

instance State (Node a)

instance Problem (Graph a) (Node a) (GraphAction a)

-- Generic Search
treeSearch :: (Problem p s a) => p -> ([s] -> (s, [s])) -> Maybe s
treeSearch problem strategy = search [startState problem]
  where
    search [] = Nothing
    search fringe
      | isGoal problem node = Just node
      | otherwise   = search (fringe' ++ expand problem node)
      where (node, fringe') = strategy fringe

graphSearch = id

