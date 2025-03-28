# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)


        "*** YOUR CODE HERE ***"
         # Initialize Q-values as a Counter (default dictionary with 0.0)
        self.qValues = util.Counter()

        # Set parameters with default values in case they are not provided
        self.alpha = args.get('alpha', 0.1)  # Learning rate
        self.discount = args.get('gamma', 0.9)  # Discount factor
        self.epsilon = args.get('epsilon', 0.1)  # Exploration probability
        self.numTraining = args.get('numTraining', 0)  # Number of training episodes

        #util.raiseNotDefined()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Return Q-value for a state-action pair if it exists, otherwise return 0.0 (default value) 
        if (state,action) in self.qValues:
            return self.qValues[(state,action)]
        else:
            return 0.0
    
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        bestValue=float("-inf") # Initialize best value as negative infinity 
        # Iterate over legal actions and return the maximum Q-value for a state 
        for action in self.getLegalActions(state):
            value=self.getQValue(state,action) # Get Q-value for a state-action pair 
            if value>bestValue:               # Update best value if a higher Q-value is found 
                bestValue=value               # Update best value 
        if bestValue==float("-inf"):         # Return 0.0 if there are no legal actions 
            return 0.0                       
        return bestValue                 # Return the best value            
    
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestValue=float("-inf")     # Initialize best value as negative infinity 
        bestAction=None             # Initialize best action as None 
        for action in self.getLegalActions(state):  # Iterate over legal actions 
            value=self.getQValue(state,action)      # Get Q-value for a state-action pair 
            if value>bestValue:             # Update best value and action if a higher Q-value is found      
                bestValue=value             # Update best value 
                bestAction=action           # Update best action 
        return bestAction                   # Return the best action 
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions)==0:        # Return None if there are no legal actions 
            return None                 
        if util.flipCoin(self.epsilon): # if the coin flip is less than epsilon, return a random action  
            action=random.choice(legalActions)
        else:                       # otherwise return the best action with probability 1-epsilon
            action=self.computeActionFromQValues(state)
        return action
    
        util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Update Q-value for a state-action pair based on the reward and the maximum Q-value for the next state 
        difference=reward+self.discount*self.computeValueFromQValues(nextState)-self.getQValue(state,action) 
        self.qValues[(state,action)]+=self.alpha*difference # Update Q-value 
        return self.qValues         # Return updated Q-values                       
    
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # Return Q-value for a state-action pair as the dot product of the weights and the feature vector 
        features=self.featExtractor.getFeatures(state,action) # Get feature vector 
        qValue=0                                          # Initialize Q-value as 0                             
        for feature in features:                      # Iterate over features and compute Q-value           
            qValue+=self.weights[feature]*features[feature]   # Compute Q-value as the dot product of the weights and the feature vector                    
        return qValue                                 # Return Q-value 
    
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Update weights based on the reward and the maximum Q-value for the next state
        difference=reward+self.discount*self.computeValueFromQValues(nextState)-self.getQValue(state,action) # Compute difference in Q-values 
        features=self.featExtractor.getFeatures(state,action) # Get feature vector 
        for feature in features:                  # Iterate over features and update weights based on the difference in Q-values 
            self.weights[feature]+=self.alpha*difference*features[feature] # Update weights 
        return self.weights                # Return updated weights                         
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            for weight in self.weights:             # Iterate over weights and print them 
                print(weight,self.weights[weight])
            pass