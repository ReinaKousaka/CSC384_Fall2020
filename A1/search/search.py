# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # OPEN is a stack of tuples in the form: (list of states, list of actions, cost)
    OPEN = util.Stack()
    OPEN.push(([problem.getStartState()], [], 0))
    while not OPEN.isEmpty():
        states, actions, cost = OPEN.pop()
        last_state = states[-1]
        if problem.isGoalState(last_state):
            return actions
        for next_state, action, step_cost in problem.getSuccessors(last_state):
            if next_state not in states:  # do path check
                OPEN.push((states + [next_state], actions + [action], cost + step_cost))
    return None

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # OPEN is a queue of tuples in the form: (list of states, list of actions, cost)
    OPEN = util.Queue()
    OPEN.push(([problem.getStartState()], [], 0))
    seen = {problem.getStartState(): 0}
    while not OPEN.isEmpty():
        states, actions, cost = OPEN.pop()
        last_state = states[-1]
        if cost <= seen[last_state]:  # expand only if this is the cheapest path to last_state
            if problem.isGoalState(last_state):
                return actions
            for next_state, action, step_cost in problem.getSuccessors(last_state):
                if next_state not in seen or cost + step_cost < seen[next_state]:  # cycle check
                    OPEN.push((states + [next_state], actions + [action], cost + step_cost))
                    seen[next_state] = step_cost + cost
    return None

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    OPEN = util.PriorityQueue()
    OPEN.push(([problem.getStartState()], [], 0), priority=0)
    seen = {problem.getStartState(): 0}
    while not OPEN.isEmpty():
        states, actions, cost = OPEN.pop()
        last_state = states[-1]
        if cost <= seen[last_state]:
            if problem.isGoalState(last_state):
                return actions
            for next_state, action, step_cost in problem.getSuccessors(last_state):
                if next_state not in seen or cost + step_cost < seen[next_state]:
                    OPEN.push((states + [next_state], actions + [action], cost + step_cost), priority=cost + step_cost)
                    seen[next_state] = step_cost + cost
    return None 

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first.
    The priority of the PriorityQueue is the function f(n) = g(n) + f(n).
    cost: g(n)
    heuristic(n, problem): h(n)
    OPEN is list of tuples storing (last_state, list of actions, cost: g(n), f(n), h(n))
    """
    OPEN = util.PriorityQueue()
    last_state = problem.getStartState()
    h = heuristic(last_state, problem)
    OPEN.push((problem.getStartState(), [], 0, h, h), priority=h)
    seen = {problem.getStartState(): 0}
    while not OPEN.isEmpty():
        # break tie, we use another temp heap to sort h(n).
        first_item = OPEN.pop()
        temp_heap = util.PriorityQueue()
        temp_heap.push(first_item, priority=first_item[4])  # index 4: h(n)
        while not OPEN.isEmpty():  # push all the items with same f(n) into the temp_heap
            temp_item = OPEN.pop()
            if temp_item[3] == first_item[3]:  # index 3: f(n)
                temp_heap.push(temp_item, priority=temp_item[4])
            else:
                OPEN.push(temp_item, priority=temp_item[3])
                break
        last_state, actions, g, f, h = temp_heap.pop()
        while not temp_heap.isEmpty():      # send all the rest items back
            temp_item = temp_heap.pop()
            OPEN.push(temp_item, priority=temp_item[3])

        if g <= seen[last_state]:
            if problem.isGoalState(last_state):
                return actions
            for next_state, action, step_cost in problem.getSuccessors(last_state):
                if next_state not in seen or g + step_cost < seen[next_state]:
                    OPEN.push((next_state, actions + [action], g + step_cost,
                               g + step_cost + heuristic(next_state, problem), heuristic(next_state, problem)),
                              priority=g + step_cost + heuristic(next_state, problem))
                    seen[next_state] = step_cost + g
    print("no path")


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
