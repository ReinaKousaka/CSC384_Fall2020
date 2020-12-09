# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        dis_to_ghost = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if dis_to_ghost <= 1:
            score -= 100000
        else:
            score += 5 * dis_to_ghost

        current_foodlist = currentGameState.getFood().asList()
        new_foodlist = newFood.asList()
        if len(current_foodlist) > len(new_foodlist):
            score += 10000
        if len(new_foodlist) > 0:
            min_dis = float('inf')
            for food in new_foodlist:
                min_dis = min(min_dis, manhattanDistance(newPos, food))
            score -= 10 * min_dis
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self._DFMiniMax(gameState, depth=0, agent=0)
        return self.action
    
    def _DFMiniMax(self, state, depth, agent=0):
        agent = agent % state.getNumAgents()
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent == 0:
            if depth == self.depth:
                return self.evaluationFunction(state)
            else:
                return self._Max(state, depth + 1, 0)
        else:
            return self._Min(state, depth, agent)
    
    def _Max(self, state, depth, agent=0):
        best_value = float('-inf')
        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action)
            temp = self._DFMiniMax(next_state, depth, agent + 1)
            if temp > best_value and depth == 1:
                self.action = action
            best_value = max(best_value, temp)
        return best_value

    def _Min(self, state, depth, agent):
        best_value = float('inf')
        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action) 
            temp = self._DFMiniMax(next_state, depth, agent + 1)
            best_value = min(best_value, temp)
        return best_value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self._AlphaBeta(gameState, depth=0, alpha=float('-inf'), beta=float('inf'), agent=0)
        return self.action
    
    def _AlphaBeta(self, state, depth, alpha, beta, agent=0):
        agent = agent % state.getNumAgents()
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent == 0:
            if depth == self.depth:
                return self.evaluationFunction(state)
            else:
                return self._Max(state, depth + 1, alpha, beta, 0)
        else:
            return self._Min(state, depth, alpha, beta, agent)
    
    def _Max(self, state, depth, alpha, beta, agent=0):
        best_value  = float('-inf')
        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action)
            temp = self._AlphaBeta(next_state, depth, alpha, beta, agent + 1)
            if temp > best_value and depth == 1:
                self.action = action
            best_value = max(best_value, temp)
            if best_value >= beta:  
                return best_value
            alpha = max(alpha, best_value)
        return best_value

    def _Min(self, state, depth, alpha, beta, agent):
        best_value = float('inf')
        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action) 
            temp = self._AlphaBeta(next_state, depth, alpha, beta, agent + 1)
            best_value = min(best_value, temp)
            if best_value <= alpha:
                return best_value
            beta = min(beta, best_value)
        return best_value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self._ExpectiMax(gameState, depth=0, agent=0)
        return self.action

    def _ExpectiMax(self, state, depth, agent=0):
        agent = agent % state.getNumAgents()
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent == 0:
            if depth == self.depth:
                return self.evaluationFunction(state)
            else:
                return self._Max(state, depth + 1, 0)
        else:
            return self._Min(state, depth, agent)
    
    def _Max(self, state, depth, agent=0):
        best_value = float('-inf')
        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action)
            temp = self._ExpectiMax(next_state, depth, agent + 1)
            if temp > best_value and depth == 1:
                self.action = action
            best_value = max(best_value, temp)
        return best_value

    def _Min(self, state, depth, agent):
        sum_value, N = 0.0, 0
        for action in state.getLegalActions(agent):
            next_state = state.generateSuccessor(agent, action) 
            sum_value += self._ExpectiMax(next_state, depth, agent + 1)
            N += 1
        return sum_value / N


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"    
    position = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    capsule = currentGameState.getCapsules()
    score = 0
    # consider food
    min_dis_food = float('inf')
    for food in food_list:
        min_dis_food = min(min_dis_food, manhattanDistance(food, position))
    score += 1.0 / min_dis_food
    # consider ghosts
    min_dis_ghost = float('inf')
    for ghost in GhostStates:
        temp = manhattanDistance(position, ghost.getPosition())
        if (min_dis_ghost > temp):
            min_dis_ghost = temp
            closestGhost = ghost
    if min_dis_ghost > 0:
        if closestGhost.scaredTimer > 0:
            score += 1.0 / min_dis_ghost
        else:
            if min_dis_ghost < 6:
                score -= 1.0 / min_dis_ghost
            if min_dis_ghost < 2:
                score -= 50
    # consider capsules
    score -= len(capsule) * 25
    if len(capsule) > 0:
        min_dis_cap = float('inf')
        for cap in capsule:
            min_dis_cap = min(min_dis_cap, manhattanDistance(cap, position))
        score -= 0.1 * min_dis_cap
    # add noise to prevent stuck
    score += random.random() * 0.1
    return currentGameState.getScore() + score
    
# Abbreviation
better = betterEvaluationFunction
