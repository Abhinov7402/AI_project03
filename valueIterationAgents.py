# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp             # The mdp we are going to work with 
        self.discount = discount        # The discount factor initialization
        self.iterations = iterations    # The number of iterations to run value iteration initialization
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()   


    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #Iterating over the number of iterations to update the values of the states in the mdp 
        for i in range(self.iterations):
            new_values = util.Counter() # Here a counter which is a dict with default 0 is used to store the new values of the states
            #Iterating over the states in the mdp to update the values of the states
            for state in self.mdp.getStates():
                #Checking if the state is a terminal state and if it is then the value of the state is set to 0 
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                #If the state is not a terminal state then the value of the state is updated by finding the maximum value of the state 
                else:
                    max_value = float("-inf")   #Initializing the maximum value of the state to negative infinity  
                    for action in self.mdp.getPossibleActions(state): #Iterating over the possible actions in the state to find the maximum value of the state 
                        value = self.computeQValueFromValues(state, action) #Computing the Q value of the state and action 
                        if value > max_value:      #Checking if the value is greater than the maximum value 
                            max_value = value       #If the value is greater than the maximum value then the maximum value is updated to the value 
                    new_values[state] = max_value   #Setting the value of the state to the maximum value 
            self.values = new_values                #Updating the values of the states in the mdp 
        



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0             #Initializing the Q value of the state and action to 0 
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action): #Iterating over the transition states and probabilities of the state and action
            #Computing the Q value of the state and action by adding the product of the probability and the reward of the state and action to the product of the discount factor and the value of the next state 
            q_value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state])
        return q_value       #Returning the Q value of the state and action 
    
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None      #Initializing the best action to None 
        best_value = float("-inf")  #Initializing the best value to negative infinity 
        for action in self.mdp.getPossibleActions(state):   #Iterating over the possible actions in the state 
            value = self.computeQValueFromValues(state, action) #Computing the Q value of the state and action 
            if value > best_value:      #Checking if the value is greater than the best value 
                best_value = value      #If the value is greater than the best value then the best value is updated to the value 
                best_action = action    #The best action is updated to the action  
        return best_action              #Returning the best action 
    
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}           #Initializing a dictionary to store the predecessors of the states 
        for state in self.mdp.getStates():  #Iterating over the states in the mdp  
            predecessors[state] = set()    #Initializing the predecessors of the state to an empty set 
        for state in self.mdp.getStates(): #Iterating over the states in the mdp 
            for action in self.mdp.getPossibleActions(state): #Iterating over the possible actions in the state  
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action): #Iterating over the transition states and probabilities of the state and action    
                    #Checking if the probability is greater than 0 and if it is then the predecessor of the next state is added to the set of predecessors of the state 
                    if prob > 0:
                        predecessors[next_state].add(state) #Adding the predecessor of the next state to the set of predecessors of the state 
        priority_queue = util.PriorityQueue() #Initializing a priority queue to store the states in the mdp 
        for state in self.mdp.getStates():  #Iterating over the states in the mdp 
            if not self.mdp.isTerminal(state): #Checking if the state is a terminal state 
                max_value = float("-inf")       #Initializing the maximum value of the state to negative infinity 
                for action in self.mdp.getPossibleActions(state): #Iterating over the possible actions in the state 
                    value = self.computeQValueFromValues(state, action) #Computing the Q value of the state and action 
                    if value > max_value:     #Checking if the value is greater than the maximum value 
                        max_value = value     #If the value is greater than the maximum value then the maximum value is updated to the value  
                diff = abs(self.values[state] - max_value) #Computing the difference between the value of the state and the maximum value 
                priority_queue.push(state, -diff) #Pushing the state and the negative difference to the priority queue 
        for i in range(self.iterations): #Iterating over the number of iterations to update the values of the states in the mdp 
            if priority_queue.isEmpty(): #Checking if the priority queue is empty and if it is then the loop is broken 
                break
            state = priority_queue.pop() #Popping the state from the priority queue 
            if not self.mdp.isTerminal(state): #Checking if the state is a terminal state
                max_value = float("-inf")       #Initializing the maximum value of the state to negative infinity 
                for action in self.mdp.getPossibleActions(state): #Iterating over the possible actions in the state 
                    value = self.computeQValueFromValues(state, action) #Computing the Q value of the state and action 
                    if value > max_value:   #Checking if the value is greater than the maximum value 
                        max_value = value      #If the value is greater than the maximum value then the maximum value is updated to the value 
                self.values[state] = max_value #Setting the value of the state to the maximum value 
            for predecessor in predecessors[state]: #Iterating over the predecessors of the state 
                max_value = float("-inf")           #Initializing the maximum value of the predecessor to negative infinity 
                for action in self.mdp.getPossibleActions(predecessor): #Iterating over the possible actions in the predecessor 
                    value = self.computeQValueFromValues(predecessor, action) #Computing the Q value of the predecessor and action 
                    if value > max_value:  #Checking if the value is greater than the maximum value 
                        max_value = value   #If the value is greater than the maximum value then the maximum value is updated to the value 
                diff = abs(self.values[predecessor] - max_value) #Computing the difference between the value of the predecessor and the maximum value 
                #Checking if the difference is greater than the threshold and if it is then the predecessor is pushed to the priority queue 
                if diff > self.theta:                            #Checking if the difference is greater than the threshold
                    priority_queue.update(predecessor, -diff)  #Pushing the predecessor to the priority queue 
        return
        util.raiseNotDefined()


