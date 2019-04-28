# Our environnment 
import pypownet.agent
import pypownet.environment

#our classifier
from sklearn.neural_network import MLPClassifier

#module to save and load the perceptron's parameters
import pickle

import numpy as np

class Submission(pypownet.agent.Agent):
    """
    An agent which load the parameters of a multi-layer perceptron previously trained, 
    in order to predict the action to take given an observations.
    The MLP predict the id of the action, which is then transformed into a numpy array, and then into an Action object.
    When the perceptron think it has a low propability to give a good answer, he calls GreedySearch.
    To ensure our agents do not calls GreedySearch to often, which would slow it down, when set a credit System.
    """

    def __init__(self, environment):
        super().__init__(environment)
        
        #Parameters of the perceptron.
        filename_clf = 'program/parameters_MLP.sav' 
        self.clf = pickle.load(open(filename_clf, 'rb'))

        #correspondence table (a numpy array), which allows to transform an action id into an action array        
        filename_U = 'program/tableauU.npy'
        self.U = np.load(filename_U)

        #When running on local, you should use :
        #filename_clf = 'example_submission/parameters_MLP.sav'
        #filename_U = 'example_submission/tableauU.npy'
        
        #We want to know how many times GreedySearch is called
        self.nb_greedy_call = 0

        #A GreedySearch instance
        self.GS_agent = pypownet.agent.GreedySearch(environment)
        self.GS_agent.verbose = False

        # The threshold above which, when the presceptron's certainty is lower, it calls GreedySearch. 
        self.threshold = 0.1

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        # observation -> observation array
        x = observation.as_array()
        
        # observation array -> action id (using the perceptron)      
        action_id = self.clf.predict(np.array([x]))[0]

        # action id -> action array (using the correspondence table)  
        action_arr = self.U[action_id]
        
        # action array -> action
        action = action_space.array_to_action(action_arr)
        
        # "certainty" : the probability with which the perceptron thinks to be right.
        certainty = np.amax(self.clf.predict_proba(np.array([x])))
                    
        # If the perceptron isn't sufficiently sure of himslef,
        # he asks the answer to the greedy search agent
        if certainty < self.threshold: 
            # the more we call the greedy search agent, 
            # the slower will be our algorithm
            
            # We ask GreedySearch what to do
            action = self.GS_agent.act(observation)
            
            self.nb_greedy_call += 1
            
            # It becomes harder to call GS
            self.threshold -= 0.1
            
            print(self.nb_greedy_call)
        
        # It becomes a little easyer to call GS       
        self.threshold += 0.01

        return action

        # No learning (i.e. self.feed_reward does pass)


