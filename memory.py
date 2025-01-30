import numpy as np

# This script contains the Memory class, which PPO uses as a replay buffer to learn from past experience

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        '''
        Generates randomized batches of experiences for training.

        Returns:    
            states: a list of states, 
            actions: a list of actions, 
            probs: a list of action probabilities, 
            values: a list of state values, 
            rewards: a list of rewards, 
            dones: a list of done booleans, 
            batches: a list of batch indices
        '''
        num_states = len(self.states)
        batch_start = np.arange(0, num_states, self.batch_size)
        indices = np.arange(num_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.values), np.array(self.rewards), np.array(self.dones), batches
    
    def store(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []