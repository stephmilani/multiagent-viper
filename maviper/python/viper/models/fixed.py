import numpy as np

class FixedAgent():
    def __init__(self, env, index, agent_type, state_representation):
        self.env = env
        self.index = index
        self.agent_type = agent_type 
        self.state_representation = state_representation
        self.action_types = {
            'stay':0, 'left':1, 'right':2, 'down':3, 'up':4
        }
        self.num_landmarks = 2
        self.num_inds = 4 
        
    def step(self, state):
        return self.get_action(state)

    def predict(self, state):
        return self.get_action(state)

    def get_action(self, state):
        action = np.zeros(self.env.action_space[self.index].n)
        if self.agent_type == 'adversary':
            action = self.get_adversary_action(state, action)
        else:
            action = self.get_agent_action(state, action)
        return np.array(action)

    def get_adversary_action(self, state, action):
        num_landmark_inds = self.num_landmarks * self.num_inds
        landmark_inds = state[:num_landmark_inds]
        landmark = np.random.choice(self.num_landmarks)
        chosen = landmark_inds[landmark*self.num_inds:(landmark+1)*self.num_inds]
        chosen_dir = np.where(chosen > 0.)[0]
        if len(chosen_dir) > 0:
            return self.get_directions(chosen_landmark, action)
        else:
            return action
        
    def get_agent_action(self, state, action):
        if self.state_representation == 'ignore_enemies':
           landmarks = [
               self.num_inds:self.num_inds+self.num_inds*self.num_landmarks
           ]
           if self.index == 1:
               landmark = landmarks[:self.num_inds]
           else:
               landmark = landmarks[self.num_inds:2*self.num_inds]
        return self.get_directions(landmark, action)

    def get_directions(self, landmark, action): 
        if landmark[0] > 0:
            action[self.action_types['right']] = 1.
        elif landmark[1] > 0:
            action[self.action_types['left']] = 1.
        elif landmark[2] > 0:
            action[self.action_types['up']] = 1.
        elif landmark[3] > 0:
            action[self.action_types['down']] = 1.
        return action 
